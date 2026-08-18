use std::ffi::CString;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchReader};
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::pyarrow::ToPyArrow;
use parqdb_local::ManagedQueryStream;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use tokio::task::AbortHandle;
use tokio_util::sync::CancellationToken;

use crate::errors::{core_error, runtime_error};

#[pyclass(name = "_NativeQueryStream", frozen)]
pub(crate) struct PyNativeQueryStream {
    stream: Arc<Mutex<Option<ManagedQueryStream>>>,
    schema: SchemaRef,
    cancellation: CancellationToken,
    runtime: Arc<Runtime>,
}

impl PyNativeQueryStream {
    pub(crate) fn new(stream: ManagedQueryStream, runtime: Arc<Runtime>) -> Self {
        let schema = stream.schema_ref();
        let cancellation = stream.cancellation_token();
        Self {
            stream: Arc::new(Mutex::new(Some(stream))),
            schema,
            cancellation,
            runtime,
        }
    }
}

#[pymethods]
impl PyNativeQueryStream {
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.schema.as_ref().to_pyarrow(py)
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        if requested_schema.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "schema projection is not supported for ParqDB query streams",
            ));
        }
        let stream = self.runtime.block_on(async {
            let mut stream = self.stream.lock().await;
            stream.take()
        });
        let stream = stream.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("query stream has already been consumed")
        })?;
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(ManagedStreamReader {
            schema: stream.schema_ref(),
            stream,
            runtime: Arc::clone(&self.runtime),
        });
        let stream = FFI_ArrowArrayStream::new(reader);
        PyCapsule::new(
            py,
            stream,
            Some(CString::new("arrow_array_stream").expect("static capsule name")),
        )
    }

    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = Arc::clone(&self.stream);
        let runtime = Arc::clone(&self.runtime);
        let mut cancel_on_drop = CancelOnDrop::new(self.cancellation.clone());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = runtime
                .spawn(async move {
                    let mut stream = stream.lock().await;
                    let Some(stream) = stream.as_mut() else {
                        return Ok(None);
                    };
                    futures::StreamExt::next(stream).await.transpose()
                })
                .await
                .map_err(|error| runtime_error(error.to_string()))?;
            match result {
                Ok(Some(batch)) => {
                    let batch = Python::attach(|py| batch.to_pyarrow(py).map(Bound::unbind))?;
                    cancel_on_drop.disarm();
                    Ok(batch)
                }
                Ok(None) => {
                    cancel_on_drop.disarm();
                    Err(PyStopAsyncIteration::new_err("stream exhausted"))
                }
                Err(error) => {
                    cancel_on_drop.disarm();
                    let error = parqdb_local::Error::from(error);
                    Err(core_error(&error))
                }
            }
        })
    }

    fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.cancellation.cancel();
        let stream = Arc::clone(&self.stream);
        let runtime = Arc::clone(&self.runtime);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            runtime
                .spawn(async move {
                    let mut stream = stream.lock().await;
                    if let Some(mut stream) = stream.take() {
                        stream.cancel();
                    }
                })
                .await
                .map_err(|error| runtime_error(error.to_string()))?;
            Ok(())
        })
    }
}

struct ManagedStreamReader {
    stream: ManagedQueryStream,
    schema: SchemaRef,
    runtime: Arc<Runtime>,
}

impl Iterator for ManagedStreamReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.runtime
            .block_on(futures::StreamExt::next(&mut self.stream))
            .map(|result| result.map_err(|error| ArrowError::ExternalError(Box::new(error))))
    }
}

impl RecordBatchReader for ManagedStreamReader {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Drop for ManagedStreamReader {
    fn drop(&mut self) {
        self.stream.cancel();
    }
}

struct CancelOnDrop(Option<CancellationToken>);

impl CancelOnDrop {
    fn new(token: CancellationToken) -> Self {
        Self(Some(token))
    }

    fn disarm(&mut self) {
        self.0.take();
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(token) = self.0.take() {
            token.cancel();
        }
    }
}

pub(crate) struct AbortOnDrop(Option<AbortHandle>);

impl AbortOnDrop {
    pub(crate) fn new(handle: AbortHandle) -> Self {
        Self(Some(handle))
    }

    pub(crate) fn disarm(&mut self) {
        self.0.take();
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

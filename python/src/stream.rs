use std::sync::Arc;

use arrow::pyarrow::ToPyArrow;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::prelude::*;
use relify_local::ManagedQueryStream;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use tokio::task::AbortHandle;
use tokio_util::sync::CancellationToken;

use crate::errors::{core_error, runtime_error};

#[pyclass(name = "_NativeQueryStream", frozen)]
pub(crate) struct PyNativeQueryStream {
    stream: Arc<Mutex<Option<ManagedQueryStream>>>,
    cancellation: CancellationToken,
    runtime: Arc<Runtime>,
}

impl PyNativeQueryStream {
    pub(crate) fn new(stream: ManagedQueryStream, runtime: Arc<Runtime>) -> Self {
        let cancellation = stream.cancellation_token();
        Self {
            stream: Arc::new(Mutex::new(Some(stream))),
            cancellation,
            runtime,
        }
    }
}

#[pymethods]
impl PyNativeQueryStream {
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
                    let error = relify_local::Error::from(error);
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

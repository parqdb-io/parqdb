use std::mem;

use arrow::array::RecordBatch;
use arrow::buffer::Buffer;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::pyarrow::{PyArrowType, ToPyArrow};
use arrow_ipc::reader::StreamDecoder;
use arrow_ipc::writer::StreamWriter;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

type PyDecodedChunk = (Option<Py<PyAny>>, Option<Py<PyAny>>, bool);

#[pyclass(name = "_IpcEncoder")]
pub(crate) struct PyIpcEncoder {
    writer: StreamWriter<Vec<u8>>,
    schema: Schema,
    max_chunk_bytes: usize,
    finished: bool,
}

#[pymethods]
impl PyIpcEncoder {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(py: Python<'_>, schema: PyArrowType<Schema>, max_chunk_bytes: usize) -> PyResult<Self> {
        validate_positive(max_chunk_bytes, "max_chunk_bytes")?;
        let writer = py
            .detach(|| StreamWriter::try_new(Vec::new(), &schema.0))
            .map_err(|error| ipc_error(&error))?;
        Ok(Self {
            writer,
            schema: schema.0,
            max_chunk_bytes,
            finished: false,
        })
    }

    fn start(&mut self, py: Python<'_>) -> Vec<Py<PyBytes>> {
        self.take_chunks(py)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn write(
        &mut self,
        py: Python<'_>,
        batch: PyArrowType<RecordBatch>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        if self.finished {
            return Err(PyValueError::new_err("Arrow IPC encoder is finished"));
        }
        if batch.0.schema().as_ref() != &self.schema {
            return Err(PyValueError::new_err(
                "record batch schema does not match the Arrow IPC stream schema",
            ));
        }
        py.detach(|| self.writer.write(&batch.0))
            .map_err(|error| ipc_error(&error))?;
        Ok(self.take_chunks(py))
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<Vec<Py<PyBytes>>> {
        if self.finished {
            return Ok(Vec::new());
        }
        py.detach(|| self.writer.finish())
            .map_err(|error| ipc_error(&error))?;
        self.finished = true;
        Ok(self.take_chunks(py))
    }
}

impl PyIpcEncoder {
    fn take_chunks(&mut self, py: Python<'_>) -> Vec<Py<PyBytes>> {
        let encoded = mem::take(self.writer.get_mut());
        encoded
            .chunks(self.max_chunk_bytes)
            .map(|chunk| PyBytes::new(py, chunk).unbind())
            .collect()
    }
}

#[pyclass(name = "_IpcDecoder")]
pub(crate) struct PyIpcDecoder {
    decoder: StreamDecoder,
    max_frame_bytes: usize,
    bytes_since_boundary: usize,
    pending: Option<Buffer>,
    schema_emitted: bool,
    finished: bool,
}

#[pymethods]
impl PyIpcDecoder {
    #[new]
    fn new(max_frame_bytes: usize) -> PyResult<Self> {
        validate_positive(max_frame_bytes, "max_frame_bytes")?;
        Ok(Self {
            decoder: StreamDecoder::new(),
            max_frame_bytes,
            bytes_since_boundary: 0,
            pending: None,
            schema_emitted: false,
            finished: false,
        })
    }

    fn push(&mut self, py: Python<'_>, chunk: &[u8]) -> PyResult<PyDecodedChunk> {
        if self.finished {
            return Err(PyValueError::new_err("Arrow IPC decoder is finished"));
        }
        let (schema, batch, has_buffered_input) = py
            .detach(|| self.decode_next(chunk))
            .map_err(|error| ipc_error(&error))?;
        let schema = schema
            .map(|schema| schema.as_ref().to_pyarrow(py).map(Bound::unbind))
            .transpose()?;
        let batch = batch
            .map(|batch| batch.to_pyarrow(py).map(Bound::unbind))
            .transpose()?;
        Ok((schema, batch, has_buffered_input))
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.finished {
            return Ok(());
        }
        if self.pending.is_some() {
            return Err(PyValueError::new_err(
                "Arrow IPC decoder has unconsumed input",
            ));
        }
        py.detach(|| self.decoder.finish())
            .map_err(|error| ipc_error(&error))?;
        self.finished = true;
        Ok(())
    }
}

impl PyIpcDecoder {
    fn decode_next(
        &mut self,
        chunk: &[u8],
    ) -> arrow::error::Result<(Option<SchemaRef>, Option<RecordBatch>, bool)> {
        if !chunk.is_empty() {
            if self.pending.is_some() {
                return Err(arrow::error::ArrowError::IpcError(
                    "buffered Arrow IPC input must be drained before pushing another chunk".into(),
                ));
            }
            self.pending = Some(Buffer::from(chunk.to_vec()));
        }
        let mut schema = None;

        while self.pending.is_some() {
            let remaining = self
                .max_frame_bytes
                .checked_sub(self.bytes_since_boundary)
                .ok_or_else(frame_too_large)?;
            if remaining == 0 {
                return Err(frame_too_large());
            }
            let input = self.pending.as_mut().expect("input is present");
            let take = remaining.min(input.len());
            let mut window = input.slice_with_length(0, take);
            let had_schema = self.decoder.schema().is_some();
            let batch = self.decoder.decode(&mut window)?;
            let consumed = take - window.len();
            input.advance(consumed);
            self.bytes_since_boundary += consumed;
            if input.is_empty() {
                self.pending = None;
            }

            if !had_schema && self.decoder.schema().is_some() {
                self.bytes_since_boundary = 0;
                if !self.schema_emitted {
                    schema = self.decoder.schema();
                    self.schema_emitted = true;
                }
            }
            if let Some(batch) = batch {
                self.bytes_since_boundary = 0;
                return Ok((schema, Some(batch), self.pending.is_some()));
            }
            if consumed == 0 {
                return Err(arrow::error::ArrowError::IpcError(
                    "Arrow IPC decoder made no progress".into(),
                ));
            }
        }

        Ok((schema, None, false))
    }
}

fn frame_too_large() -> arrow::error::ArrowError {
    arrow::error::ArrowError::IpcError(
        "Arrow IPC frame exceeds the configured max_frame_bytes".into(),
    )
}

fn validate_positive(value: usize, name: &str) -> PyResult<()> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be positive")));
    }
    Ok(())
}

fn ipc_error(error: &arrow::error::ArrowError) -> PyErr {
    PyValueError::new_err(format!("invalid Arrow IPC stream: {error}"))
}

pub(crate) fn add_ipc_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIpcEncoder>()?;
    module.add_class::<PyIpcDecoder>()?;
    Ok(())
}

//! `PyO3` bindings for the Relify Python package.

mod datafusion;
mod errors;
mod index;
mod session;
mod stream;

use pyo3::prelude::*;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::add_exceptions(module)?;
    datafusion::add_datafusion_bindings(module)?;
    index::add_index_bindings(module)?;
    session::add_session_bindings(module)?;
    module.add_class::<stream::PyNativeQueryStream>()?;
    Ok(())
}

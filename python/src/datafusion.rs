use std::collections::HashSet;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyType};

pub(crate) fn add_datafusion_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    install_datafusion_bindings(module)
}

fn install_datafusion_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let internal = PyModule::new(py, "parqdb.datafusion._internal")?;
    datafusion_python::init_internal_module(py, &internal)?;
    relocate_type_modules(&internal, &mut HashSet::new())?;

    let sys = PyModule::import(py, "sys")?;
    let modules = sys.getattr("modules")?.cast_into::<PyDict>()?;
    modules.set_item("parqdb.datafusion._internal", &internal)?;

    // Keep an owning reference in the extension module as well as sys.modules.
    module.add("_datafusion_internal", internal)?;
    Ok(())
}

fn relocate_type_modules(
    module: &Bound<'_, PyModule>,
    visited: &mut HashSet<usize>,
) -> PyResult<()> {
    if !visited.insert(module.as_ptr() as usize) {
        return Ok(());
    }

    for (_, value) in module.dict().iter() {
        if let Ok(class) = value.cast::<PyType>() {
            let module_name = class.getattr("__module__")?.extract::<String>()?;
            if module_name == "datafusion" || module_name.starts_with("datafusion.") {
                class.setattr("__module__", format!("parqdb.{}", module_name.as_str()))?;
            }
        }
        if let Ok(child) = value.cast::<PyModule>() {
            relocate_type_modules(child, visited)?;
        }
    }
    Ok(())
}

use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use relify_catalog::Error as CatalogErrorKind;
use relify_index::Error as IndexError;
use relify_local::Error as LocalError;

create_exception!(
    _native,
    RelifyError,
    PyRuntimeError,
    "Base class for errors reported by Relify."
);
create_exception!(
    _native,
    InvalidArgumentError,
    RelifyError,
    "A Relify operation received an invalid argument."
);
create_exception!(
    _native,
    InvalidSchemaError,
    RelifyError,
    "A source or index table has an invalid schema or value."
);
create_exception!(
    _native,
    IndexNotFoundError,
    RelifyError,
    "The requested index does not exist."
);
create_exception!(
    _native,
    AlreadyExistsError,
    RelifyError,
    "The requested index already exists."
);
create_exception!(
    _native,
    BuildAlreadyRunningError,
    RelifyError,
    "Another process is already building the requested index."
);
create_exception!(
    _native,
    AmbiguousIndexError,
    RelifyError,
    "More than one index matches the request."
);
create_exception!(
    _native,
    InvalidMetadataError,
    RelifyError,
    "An index metadata document is invalid."
);
create_exception!(
    _native,
    CatalogError,
    RelifyError,
    "An index catalog operation failed."
);
create_exception!(
    _native,
    StorageError,
    RelifyError,
    "A storage operation failed."
);
create_exception!(
    _native,
    BackendError,
    RelifyError,
    "A query backend operation failed."
);
create_exception!(
    _native,
    QueryQueueFullError,
    RelifyError,
    "The bounded query queue has no free entry."
);
create_exception!(
    _native,
    QueryQueueTimeoutError,
    RelifyError,
    "A query timed out while waiting for runtime admission."
);

pub(crate) fn add_exceptions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("RelifyError", py.get_type::<RelifyError>())?;
    module.add(
        "InvalidArgumentError",
        py.get_type::<InvalidArgumentError>(),
    )?;
    module.add("InvalidSchemaError", py.get_type::<InvalidSchemaError>())?;
    module.add("IndexNotFoundError", py.get_type::<IndexNotFoundError>())?;
    module.add("AlreadyExistsError", py.get_type::<AlreadyExistsError>())?;
    module.add(
        "BuildAlreadyRunningError",
        py.get_type::<BuildAlreadyRunningError>(),
    )?;
    module.add("AmbiguousIndexError", py.get_type::<AmbiguousIndexError>())?;
    module.add(
        "InvalidMetadataError",
        py.get_type::<InvalidMetadataError>(),
    )?;
    module.add("CatalogError", py.get_type::<CatalogError>())?;
    module.add("StorageError", py.get_type::<StorageError>())?;
    module.add("BackendError", py.get_type::<BackendError>())?;
    module.add("QueryQueueFullError", py.get_type::<QueryQueueFullError>())?;
    module.add(
        "QueryQueueTimeoutError",
        py.get_type::<QueryQueueTimeoutError>(),
    )?;
    Ok(())
}

pub(crate) fn core_error(error: &LocalError) -> PyErr {
    let message = error.to_string();
    match error {
        LocalError::InvalidArgument(_) => InvalidArgumentError::new_err(message),
        LocalError::InvalidSchema(_) => InvalidSchemaError::new_err(message),
        LocalError::IndexNotFound(_) => IndexNotFoundError::new_err(message),
        LocalError::AlreadyExists(_) => AlreadyExistsError::new_err(message),
        LocalError::BuildAlreadyRunning(_) => BuildAlreadyRunningError::new_err(message),
        LocalError::QueryQueueFull(_) => QueryQueueFullError::new_err(message),
        LocalError::QueryQueueTimeout(_) => QueryQueueTimeoutError::new_err(message),
        LocalError::AmbiguousIndex(_) => AmbiguousIndexError::new_err(message),
        LocalError::InvalidMetadata(_) => InvalidMetadataError::new_err(message),
        LocalError::Catalog(error) => catalog_error(error, message),
        LocalError::Storage(_)
        | LocalError::Io(_)
        | LocalError::Json(_)
        | LocalError::Parquet(_) => StorageError::new_err(message),
        LocalError::DataFusion(_)
        | LocalError::Arrow(_)
        | LocalError::Kernel(_)
        | LocalError::Iceberg(_) => BackendError::new_err(message),
    }
}

pub(crate) fn index_error(error: &IndexError) -> PyErr {
    let message = error.to_string();
    match error {
        IndexError::InvalidMetadata(_) => InvalidMetadataError::new_err(message),
        IndexError::InvalidTimestamp(_) => InvalidArgumentError::new_err(message),
        IndexError::IndexNotFound(_) => IndexNotFoundError::new_err(message),
        IndexError::AmbiguousIndex(_) => AmbiguousIndexError::new_err(message),
        IndexError::Catalog(error) => catalog_error(error, message),
        IndexError::Storage(_) | IndexError::Json(_) => StorageError::new_err(message),
        _ => RelifyError::new_err(message),
    }
}

fn catalog_error(error: &CatalogErrorKind, message: String) -> PyErr {
    match error {
        CatalogErrorKind::InvalidIdentifier(_) => InvalidArgumentError::new_err(message),
        CatalogErrorKind::IndexNotFound(_) => IndexNotFoundError::new_err(message),
        CatalogErrorKind::AlreadyExists(_) => AlreadyExistsError::new_err(message),
        CatalogErrorKind::InvalidMetadata(_) => InvalidMetadataError::new_err(message),
        CatalogErrorKind::Io(_) | CatalogErrorKind::Json(_) => StorageError::new_err(message),
        _ => CatalogError::new_err(message),
    }
}

pub(crate) fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    RelifyError::new_err(error.to_string())
}

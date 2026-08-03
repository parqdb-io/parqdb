from importlib.metadata import version as _package_version

from . import _native as _native
from . import backends as backends
from . import builders as builders
from . import datafusion as datafusion
from . import datasets as datasets
from . import experimental as experimental
from ._native import (
    AlreadyExistsError,
    AmbiguousIndexError,
    BackendError,
    BuildAlreadyRunningError,
    CatalogError,
    IndexNotFoundError,
    InvalidArgumentError,
    InvalidMetadataError,
    InvalidSchemaError,
    RelifyError,
    StorageError,
)
from .api import (
    IVF,
    CatalogEntry,
    IndexCacheInfo,
    IndexCatalog,
    IndexInfo,
    IndexStatus,
    Local,
    Maintenance,
    MaintenanceObject,
    Session,
    SourceTable,
    Table,
    TableIdentifier,
    VectorQuery,
    WriteOptions,
    connect,
    open_index_catalog,
)

__version__ = _package_version("relify")

__all__ = [
    "IVF",
    "AlreadyExistsError",
    "AmbiguousIndexError",
    "BackendError",
    "BuildAlreadyRunningError",
    "CatalogEntry",
    "CatalogError",
    "IndexCacheInfo",
    "IndexCatalog",
    "IndexInfo",
    "IndexNotFoundError",
    "IndexStatus",
    "InvalidArgumentError",
    "InvalidMetadataError",
    "InvalidSchemaError",
    "Local",
    "Maintenance",
    "MaintenanceObject",
    "RelifyError",
    "Session",
    "SourceTable",
    "StorageError",
    "Table",
    "TableIdentifier",
    "VectorQuery",
    "WriteOptions",
    "__version__",
    "backends",
    "builders",
    "connect",
    "datafusion",
    "datasets",
    "experimental",
    "open_index_catalog",
]

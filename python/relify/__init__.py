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
    AsyncSession,
    AsyncSourceTable,
    CatalogEntry,
    IndexCatalog,
    IndexInfo,
    IndexStatus,
    Local,
    Maintenance,
    MaintenanceObject,
    ParquetPageCacheStats,
    Session,
    SessionConfig,
    SourceTable,
    Table,
    TableIdentifier,
    VectorQuery,
    WriteOptions,
    connect,
    connect_async,
    open_index_catalog,
)

__version__ = _package_version("relify")

__all__ = [
    "IVF",
    "AlreadyExistsError",
    "AmbiguousIndexError",
    "AsyncSession",
    "AsyncSourceTable",
    "BackendError",
    "BuildAlreadyRunningError",
    "CatalogEntry",
    "CatalogError",
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
    "ParquetPageCacheStats",
    "RelifyError",
    "Session",
    "SessionConfig",
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
    "connect_async",
    "datafusion",
    "datasets",
    "experimental",
    "open_index_catalog",
]

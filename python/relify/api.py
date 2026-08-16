"""Compatibility imports for Relify's public Python API."""

from ._native import (
    BuildAlreadyRunningError,
    QueryQueueFullError,
    QueryQueueTimeoutError,
)
from .build import IndexStatus
from .config import IVF, SessionConfig, WriteOptions
from .errors import (
    ServiceUnavailableError,
    StreamExecutionError,
    UnsupportedOperationError,
)
from .facade import (
    AsyncSession,
    AsyncSourceTable,
    Session,
    SourceTable,
    connect,
    connect_async,
)
from .identifier import TableIdentifier
from .maintenance import Maintenance, MaintenanceObject
from .query import VectorQuery
from .runtime.catalog import IndexInfo
from .session import ParquetPageCacheStats
from .table import Table

__all__ = [
    "IVF",
    "AsyncSession",
    "AsyncSourceTable",
    "BuildAlreadyRunningError",
    "IndexInfo",
    "IndexStatus",
    "Maintenance",
    "MaintenanceObject",
    "ParquetPageCacheStats",
    "QueryQueueFullError",
    "QueryQueueTimeoutError",
    "ServiceUnavailableError",
    "Session",
    "SessionConfig",
    "SourceTable",
    "StreamExecutionError",
    "Table",
    "TableIdentifier",
    "UnsupportedOperationError",
    "VectorQuery",
    "WriteOptions",
    "connect",
    "connect_async",
]

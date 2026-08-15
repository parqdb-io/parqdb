"""Compatibility imports for Relify's public Python API."""

from ._native import BuildAlreadyRunningError
from .build import IndexStatus
from .catalog import CatalogEntry, IndexCatalog, IndexInfo, open_index_catalog
from .config import IVF, SessionConfig, WriteOptions
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
from .session import ParquetPageCacheStats
from .table import Table

__all__ = [
    "IVF",
    "AsyncSession",
    "AsyncSourceTable",
    "BuildAlreadyRunningError",
    "CatalogEntry",
    "IndexCatalog",
    "IndexInfo",
    "IndexStatus",
    "Maintenance",
    "MaintenanceObject",
    "ParquetPageCacheStats",
    "Session",
    "SessionConfig",
    "SourceTable",
    "Table",
    "TableIdentifier",
    "VectorQuery",
    "WriteOptions",
    "connect",
    "connect_async",
    "open_index_catalog",
]

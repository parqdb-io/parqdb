"""Compatibility imports for Relify's public Python API."""

from ._native import BuildAlreadyRunningError
from .build import IndexStatus
from .catalog import CatalogEntry, IndexCatalog, IndexInfo, open_index_catalog
from .config import IVF, Local, SessionConfig, WriteOptions
from .identifier import TableIdentifier
from .maintenance import Maintenance, MaintenanceObject
from .query import VectorQuery
from .session import (
    ParquetPageCacheStats,
    Session,
    SourceTable,
    connect,
)
from .table import Table

__all__ = [
    "IVF",
    "BuildAlreadyRunningError",
    "CatalogEntry",
    "IndexCatalog",
    "IndexInfo",
    "IndexStatus",
    "Local",
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
    "open_index_catalog",
]

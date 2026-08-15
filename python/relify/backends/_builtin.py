from __future__ import annotations

from collections.abc import Callable

from ._capabilities import optional_iceberg_report
from .v1 import (
    BackendCapabilities,
    BackendInfo,
    CapabilityReport,
    MaintenanceOperation,
    QueryProfile,
    SimpleBackendPlugin,
    Terminal,
)

IVF_PARQUET_PARQUET = QueryProfile("ivf", "parquet", "parquet")
IVF_ICEBERG_ICEBERG = QueryProfile("ivf", "iceberg", "iceberg")

LOCAL_CAPABILITIES = BackendCapabilities(
    query_profiles=frozenset(
        {
            IVF_PARQUET_PARQUET,
            IVF_ICEBERG_ICEBERG,
        }
    ),
    terminals=frozenset(
        {
            Terminal.COLLECT,
            Terminal.SQL,
            Terminal.EXPLAIN,
            Terminal.ANALYZE,
        }
    ),
    maintenance=frozenset(
        {
            MaintenanceOperation.DROP,
            MaintenanceOperation.CACHE,
            MaintenanceOperation.GC,
        }
    ),
)

LOCAL_INFO = BackendInfo("local", "Local DataFusion", "relify")


def local_report(*, iceberg: bool) -> CapabilityReport:
    return optional_iceberg_report(LOCAL_CAPABILITIES, iceberg=iceberg)


def load_local_plugin() -> SimpleBackendPlugin:
    from ..facade import connect

    return SimpleBackendPlugin(LOCAL_INFO, LOCAL_CAPABILITIES, connect)


BUILTIN_LOADERS: dict[str, Callable[[], SimpleBackendPlugin]] = {
    "local": load_local_plugin,
}

BUILTIN_INFO = {LOCAL_INFO.name: LOCAL_INFO}

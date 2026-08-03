from __future__ import annotations

from ...backends.v1 import (
    BackendCapabilities,
    BackendInfo,
    CapabilityReport,
    MaintenanceOperation,
    QueryProfile,
    Terminal,
)

STARROCKS_CAPABILITIES = BackendCapabilities(
    query_profiles=frozenset({QueryProfile("ivf", "iceberg", "iceberg")}),
    terminals=frozenset(
        {
            Terminal.COLLECT,
            Terminal.SQL,
            Terminal.EXPLAIN,
        }
    ),
    maintenance=frozenset({MaintenanceOperation.DROP}),
)
STARROCKS_INFO = BackendInfo("starrocks", "StarRocks", "relify")


def starrocks_report() -> CapabilityReport:
    return CapabilityReport.fully_available(STARROCKS_CAPABILITIES)

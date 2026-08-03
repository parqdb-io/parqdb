from __future__ import annotations

from ...backends._capabilities import optional_iceberg_report
from ...backends.v1 import (
    BackendCapabilities,
    BackendInfo,
    CapabilityReport,
    MaintenanceOperation,
    QueryProfile,
    Terminal,
)

SPARK_CAPABILITIES = BackendCapabilities(
    query_profiles=frozenset(
        {
            QueryProfile("ivf", "parquet", "parquet"),
            QueryProfile("ivf", "iceberg", "iceberg"),
        }
    ),
    terminals=frozenset(
        {
            Terminal.COLLECT,
            Terminal.DATAFRAME,
            Terminal.EXPLAIN,
        }
    ),
    maintenance=frozenset({MaintenanceOperation.DROP}),
)
SPARK_INFO = BackendInfo("spark", "Apache Spark", "relify")


def spark_report(*, iceberg: bool) -> CapabilityReport:
    return optional_iceberg_report(SPARK_CAPABILITIES, iceberg=iceberg)

from __future__ import annotations

from .v1 import BackendCapabilities, Capability, CapabilityReport


def optional_iceberg_report(
    declared: BackendCapabilities,
    *,
    iceberg: bool,
) -> CapabilityReport:
    if iceberg:
        return CapabilityReport.fully_available(declared)
    available_query = frozenset(
        profile
        for profile in declared.query_profiles
        if profile.source_profile != "iceberg" and profile.index_profile != "iceberg"
    )
    available = BackendCapabilities(
        query_profiles=available_query,
        terminals=declared.terminals,
        maintenance=declared.maintenance,
        extensions=declared.extensions,
    )
    unavailable: dict[Capability, str] = {
        capability: "the session has no Iceberg catalog"
        for capability in declared.all() - available.all()
    }
    return CapabilityReport(declared, available, unavailable)

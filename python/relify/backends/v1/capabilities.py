from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias


class CapabilityStatus(StrEnum):
    """Availability of one declared backend capability."""

    SUPPORTED = "supported"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


class Terminal(StrEnum):
    """Public query terminals exposed by a backend session."""

    COLLECT = "collect"
    DATAFRAME = "dataframe"
    RELATION = "relation"
    SQL = "sql"
    EXPLAIN = "explain"
    ANALYZE = "analyze"


class MaintenanceOperation(StrEnum):
    """Optional index-maintenance operations exposed by a backend."""

    DROP = "drop"
    CACHE = "cache"
    GC = "gc"


@dataclass(frozen=True, order=True, slots=True)
class QueryProfile:
    """One supported index/source/index-table query combination."""

    family: str
    source_profile: str
    index_profile: str

    def __post_init__(self) -> None:
        _validate_name(self.family, "family")
        _validate_name(self.source_profile, "source_profile")
        _validate_name(self.index_profile, "index_profile")


Capability: TypeAlias = QueryProfile | Terminal | MaintenanceOperation


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Typed upper bound or available subset of backend capabilities."""

    query_profiles: frozenset[QueryProfile] = field(default_factory=frozenset)
    terminals: frozenset[Terminal] = field(default_factory=frozenset)
    maintenance: frozenset[MaintenanceOperation] = field(default_factory=frozenset)
    extensions: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_profiles", frozenset(self.query_profiles))
        object.__setattr__(self, "terminals", frozenset(self.terminals))
        object.__setattr__(self, "maintenance", frozenset(self.maintenance))
        if self.query_profiles and Terminal.COLLECT not in self.terminals:
            raise ValueError("query profiles require the collect terminal")
        extensions = dict(self.extensions)
        for name in extensions:
            _validate_name(name, "extension name")
        object.__setattr__(self, "extensions", MappingProxyType(extensions))

    def supports(self, capability: Capability) -> bool:
        if isinstance(capability, QueryProfile):
            return capability in self.query_profiles
        if isinstance(capability, Terminal):
            return capability in self.terminals
        if isinstance(capability, MaintenanceOperation):
            return capability in self.maintenance
        raise TypeError(f"unsupported capability type: {type(capability).__name__}")

    def is_subset_of(self, other: BackendCapabilities) -> bool:
        return (
            self.query_profiles <= other.query_profiles
            and self.terminals <= other.terminals
            and self.maintenance <= other.maintenance
            and self.extensions.keys() <= other.extensions.keys()
        )

    def all(self) -> frozenset[Capability]:
        return frozenset(
            (
                *self.query_profiles,
                *self.terminals,
                *self.maintenance,
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible capability inventory."""
        return {
            "query_profiles": [
                _profile_dict(profile) for profile in sorted(self.query_profiles)
            ],
            "terminals": sorted(terminal.value for terminal in self.terminals),
            "maintenance": sorted(operation.value for operation in self.maintenance),
            "extensions": dict(self.extensions),
        }


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """Declared and currently available capabilities for one bound session."""

    declared: BackendCapabilities
    available: BackendCapabilities
    unavailable: Mapping[Capability, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.available.is_subset_of(self.declared):
            raise ValueError("available capabilities must be declared")
        unavailable = dict(self.unavailable)
        for capability, reason in unavailable.items():
            if not self.declared.supports(capability):
                raise ValueError("unavailable capability must be declared")
            if self.available.supports(capability):
                raise ValueError("supported capability cannot be unavailable")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("unavailable capability reason must not be empty")
        missing = self.declared.all() - self.available.all()
        if missing != unavailable.keys():
            raise ValueError(
                "every declared but unavailable capability requires one reason"
            )
        object.__setattr__(self, "unavailable", MappingProxyType(unavailable))

    @classmethod
    def fully_available(
        cls,
        capabilities: BackendCapabilities,
    ) -> CapabilityReport:
        return cls(capabilities, capabilities)

    def status(self, capability: Capability) -> CapabilityStatus:
        if self.available.supports(capability):
            return CapabilityStatus.SUPPORTED
        if self.declared.supports(capability):
            return CapabilityStatus.UNAVAILABLE
        return CapabilityStatus.UNSUPPORTED

    def reason(self, capability: Capability) -> str | None:
        return self.unavailable.get(capability)

    def supports(self, capability: Capability) -> bool:
        return self.status(capability) is CapabilityStatus.SUPPORTED

    def to_dict(self) -> dict[str, object]:
        """Return declared, available, and unavailable capability inventories."""
        return {
            "declared": self.declared.to_dict(),
            "available": self.available.to_dict(),
            "unavailable": [
                {
                    "capability": _capability_dict(capability),
                    "reason": reason,
                }
                for capability, reason in sorted(
                    self.unavailable.items(),
                    key=lambda item: repr(item[0]),
                )
            ],
        }


def _validate_name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _profile_dict(profile: QueryProfile) -> dict[str, str]:
    return {
        "family": profile.family,
        "source_profile": profile.source_profile,
        "index_profile": profile.index_profile,
    }


def _capability_dict(capability: Capability) -> dict[str, object]:
    if isinstance(capability, QueryProfile):
        return {"kind": "query_profile", **_profile_dict(capability)}
    if isinstance(capability, Terminal):
        return {"kind": "terminal", "name": capability.value}
    if isinstance(capability, MaintenanceOperation):
        return {"kind": "maintenance", "name": capability.value}
    raise TypeError(f"unsupported capability type: {type(capability).__name__}")

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..identifier import TableIdentifier

BUILDER_API_VERSION = 1


@dataclass(frozen=True, order=True, slots=True)
class BuildProfile:
    """One source/index-table combination supported by an index builder."""

    family: str
    source_profile: str
    index_profile: str

    def __post_init__(self) -> None:
        for name, value in (
            ("family", self.family),
            ("source_profile", self.source_profile),
            ("index_profile", self.index_profile),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BuilderCapabilities:
    """Typed capabilities advertised by one index builder."""

    profiles: frozenset[BuildProfile] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        profiles = frozenset(self.profiles)
        if any(not isinstance(profile, BuildProfile) for profile in profiles):
            raise TypeError("builder profiles must be BuildProfile values")
        object.__setattr__(self, "profiles", profiles)

    def supports(self, profile: BuildProfile) -> bool:
        return profile in self.profiles

    def to_dict(self) -> dict[str, object]:
        return {
            "profiles": [
                {
                    "family": profile.family,
                    "source_profile": profile.source_profile,
                    "index_profile": profile.index_profile,
                }
                for profile in sorted(self.profiles)
            ]
        }


@dataclass(frozen=True, slots=True)
class BuilderInfo:
    """Identity and compatibility information for one builder implementation."""

    name: str
    display_name: str
    distribution: str
    api_version: int = BUILDER_API_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not self.name.isascii()
            or self.name.lower() != self.name
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in self.name
            )
        ):
            raise ValueError(
                "builder name must contain lowercase ASCII letters, digits, '_' or '-'"
            )
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("builder display_name must not be empty")
        if not isinstance(self.distribution, str) or not self.distribution.strip():
            raise ValueError("builder distribution must not be empty")
        if not isinstance(self.api_version, int) or isinstance(
            self.api_version,
            bool,
        ):
            raise TypeError("builder api_version must be an integer")
        if self.api_version <= 0:
            raise ValueError("builder api_version must be positive")


@dataclass(frozen=True, slots=True)
class BuildRequest:
    """Pinned, backend-independent inputs for one initial index build."""

    source_identifier: TableIdentifier
    source: Mapping[str, object]
    index: str
    column: str
    key: tuple[str, ...]
    config: Any
    writer_options: Any

    def __post_init__(self) -> None:
        if not isinstance(self.source_identifier, TableIdentifier):
            raise TypeError("source_identifier must be relify.TableIdentifier")
        if not isinstance(self.source, Mapping):
            raise TypeError("source must be a portable relation mapping")
        if not isinstance(self.index, str) or not self.index:
            raise ValueError("index must be a non-empty string")
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("column must be a non-empty string")
        key = tuple(self.key)
        if (
            not key
            or any(not isinstance(field, str) or not field for field in key)
            or len(set(key)) != len(key)
        ):
            raise ValueError("key must contain unique, non-empty strings")
        object.__setattr__(self, "source", _freeze_mapping(self.source))
        object.__setattr__(self, "key", key)
        _ = self.profile

    @property
    def profile(self) -> str:
        profile = self.source.get("profile")
        if not isinstance(profile, str) or not profile:
            raise ValueError("source relation has no valid profile")
        return profile


@dataclass(frozen=True, slots=True)
class BuildProgressSnapshot:
    """One point-in-time view of an index build's progress."""

    phase: str | None
    completed: int | None
    total: int | None
    fraction: float | None


@runtime_checkable
class BuildProgress(Protocol):
    """Progress source observed by the asynchronous build coordinator."""

    def snapshot(self) -> BuildProgressSnapshot: ...


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Catalog and runtime bindings supplied to a builder for one session."""

    iceberg_catalog: Any | None = field(default=None, repr=False)
    catalog_name: str | None = None
    index_namespace: tuple[str, ...] = ("relify",)
    progress: BuildProgress | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        namespace = tuple(self.index_namespace)
        if not namespace or any(
            not isinstance(segment, str) or not segment for segment in namespace
        ):
            raise ValueError("index_namespace must contain non-empty strings")
        if self.catalog_name is not None and (
            not isinstance(self.catalog_name, str) or not self.catalog_name
        ):
            raise ValueError("catalog_name must be a non-empty string")
        if self.progress is not None and not isinstance(self.progress, BuildProgress):
            raise TypeError("progress must implement relify.builders.BuildProgress")
        object.__setattr__(self, "index_namespace", namespace)


class BuildResult:
    """Marker for a completed builder result consumed by the coordinator."""


@dataclass(frozen=True, slots=True)
class BuildOutput(BuildResult):
    """Validated index relations and parameters ready for publication.

    A builder must finish writing and validate each relation's physical schema
    before returning. The coordinator may publish this output immediately.
    """

    parameters: Mapping[str, str]
    index_relations: Mapping[str, Mapping[str, object]]
    discard: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        parameters = {str(name): str(value) for name, value in self.parameters.items()}
        relations = {
            str(role): _freeze_mapping(reference)
            for role, reference in self.index_relations.items()
        }
        if not parameters:
            raise ValueError("build output parameters must not be empty")
        if not relations:
            raise ValueError("build output relations must not be empty")
        if any(not name for name in parameters):
            raise ValueError("build output parameter names must not be empty")
        if any(not role for role in relations):
            raise ValueError("build output relation roles must not be empty")
        for reference in relations.values():
            profile = reference.get("profile")
            if not isinstance(profile, str) or not profile:
                raise ValueError("build output relation has no valid profile")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(
            self,
            "index_relations",
            MappingProxyType(relations),
        )


@runtime_checkable
class IndexBuilder(Protocol):
    """Public extension contract for an index construction engine."""

    @property
    def info(self) -> BuilderInfo: ...

    @property
    def capabilities(self) -> BuilderCapabilities: ...

    def build(self, request: BuildRequest, context: BuildContext) -> BuildResult: ...


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(name): _freeze_json(child) for name, child in value.items()}
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(_freeze_json(child) for child in value)
    return value

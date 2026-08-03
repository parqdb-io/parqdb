from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol


class _CatalogNative(Protocol):
    def list_indexes(self) -> list[str]: ...

    def load_index_entry(self, name: str) -> tuple[str, str]: ...

    def register_index(self, name: str, metadata_location: str) -> None: ...

    def drop_index(self, name: str) -> None: ...

    def list_source_indexes(
        self,
        source: str,
        /,
    ) -> list[tuple[str, str, str, str, Mapping[str, str], int]]: ...

    def select_index(
        self,
        source: str,
        /,
        index: str | None = ...,
        column: str | None = ...,
    ) -> tuple[str, str, str]: ...


@dataclass(frozen=True)
class CatalogEntry:
    identifier: str
    metadata_location: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class IndexInfo:
    name: str
    column: str
    family: str
    metric: str
    parameters: Mapping[str, str]
    current_snapshot_id: int


class IndexCatalog:
    def __init__(self, native: _CatalogNative) -> None:
        self._native = native

    def list(self) -> list[str]:
        return self._native.list_indexes()

    def load(self, index: str) -> CatalogEntry:
        metadata_location, metadata = self._native.load_index_entry(index)
        return CatalogEntry(
            identifier=index,
            metadata_location=metadata_location,
            metadata=freeze_json(json.loads(metadata)),
        )

    def register(self, index: str, metadata_location: str) -> None:
        self._native.register_index(index, metadata_location)

    def drop(self, index: str) -> None:
        self._native.drop_index(index)

    def list_for(self, source: Mapping[str, Any]) -> list[IndexInfo]:
        """List indexes bound to one exact portable source relation."""
        return [
            IndexInfo(
                name=name,
                column=column,
                family=family,
                metric=metric,
                parameters=MappingProxyType(dict(parameters)),
                current_snapshot_id=current_snapshot_id,
            )
            for (
                name,
                column,
                family,
                metric,
                parameters,
                current_snapshot_id,
            ) in self._native.list_source_indexes(_relation_json(source))
        ]

    def select(
        self,
        source: Mapping[str, Any],
        *,
        index: str | None = None,
        column: str | None = None,
    ) -> CatalogEntry:
        """Select and load one index for an exact source relation."""
        identifier, metadata_location, metadata = self._native.select_index(
            _relation_json(source),
            index,
            column,
        )
        return CatalogEntry(
            identifier=identifier,
            metadata_location=metadata_location,
            metadata=freeze_json(json.loads(metadata)),
        )


def open_index_catalog(
    catalog: str,
    *,
    metadata_root: str | None = None,
    storage_options: Mapping[str, str] | None = None,
) -> IndexCatalog:
    """Open the query-engine-independent Relify index catalog."""
    from ._repository import open_index_repository

    return IndexCatalog(
        open_index_repository(
            catalog,
            metadata_root=metadata_root,
            storage_options=storage_options,
        )
    )


def freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(freeze_json(child) for child in value)
    return value


def _relation_json(source: Mapping[str, Any]) -> str:
    if not isinstance(source, Mapping):
        raise TypeError("source must be a portable relation mapping")
    return json.dumps(
        _mutable_json(source),
        separators=(",", ":"),
        sort_keys=True,
    )


def _mutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json(child) for child in value]
    return value

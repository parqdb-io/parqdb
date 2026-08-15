from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ...identifier import TableIdentifier
from ...query import VectorQuery


@dataclass(frozen=True, slots=True)
class ResolvedExactSearch:
    """Validated exact-search intent ready for backend compilation."""

    source: TableIdentifier
    source_relation: Mapping[str, Any]
    query_vector: tuple[float, ...]
    vector_field: str
    projection: tuple[str, ...]
    predicate: str | None
    limit: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_relation", _freeze_relation(self.source_relation)
        )


@dataclass(frozen=True, slots=True)
class ResolvedSearch:
    """Validated indexed-search semantics shared by backend compilers."""

    source: TableIdentifier
    source_relation: Mapping[str, Any]
    index: str
    centroids_relation: Mapping[str, Any]
    postings_relation: Mapping[str, Any]
    query_vector: tuple[float, ...]
    projection: tuple[str, ...]
    predicate: str | None
    limit: int
    dimension: int
    nlist: int
    nprobe: int
    source_key_fields: tuple[str, ...]
    vector_field: str
    posting_encoding: str
    needs_source: bool
    snapshot_id: int
    family: str
    index_schema_version: int
    metric: str
    ntotal: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_relation", _freeze_relation(self.source_relation)
        )
        object.__setattr__(
            self,
            "centroids_relation",
            _freeze_relation(self.centroids_relation),
        )
        object.__setattr__(
            self,
            "postings_relation",
            _freeze_relation(self.postings_relation),
        )


def validate_query(query: VectorQuery) -> None:
    """Validate backend-independent vector-query inputs."""
    if not isinstance(query, VectorQuery):
        raise TypeError("query must be a relify.VectorQuery")
    if not query.query:
        raise ValueError("query vector must not be empty")
    for value in query.query:
        if not math.isfinite(value):
            raise ValueError("query vector must contain finite float values")
        try:
            narrowed = struct.unpack("!f", struct.pack("!f", value))[0]
        except OverflowError as error:
            raise ValueError("query vector must contain finite float values") from error
        if not math.isfinite(narrowed):
            raise ValueError("query vector must contain finite float values")
    if query.result_limit <= 0:
        raise ValueError("limit must be positive")


def resolve_projection(
    source_fields: Sequence[str],
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve and validate the portable source-field projection."""
    fields = tuple(source_fields)
    if "_distance" in fields:
        raise ValueError("source table must not contain reserved column _distance")
    if requested is None:
        return fields
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("projection must contain unique source fields")
    missing = [field for field in requested if field not in fields]
    if missing:
        raise ValueError(f"projection contains unknown source fields: {missing}")
    return requested


def resolve_vector_field(
    *,
    requested: str | None,
    source_fields: Sequence[str],
    vector_fields: Sequence[str],
) -> str:
    """Resolve a requested or unambiguous backend-described vector field."""
    if requested is not None:
        if requested not in source_fields:
            raise ValueError(f"unknown vector column: {requested}")
        if requested not in vector_fields:
            raise TypeError("vector column must be a list<float> or list<double> field")
        return requested
    candidates = tuple(vector_fields)
    if len(candidates) != 1:
        raise ValueError(
            "column is required unless the source has one list<float> or "
            "list<double> field"
        )
    return candidates[0]


def resolve_exact_search(
    query: VectorQuery,
    *,
    source_relation: Mapping[str, Any],
    projection: tuple[str, ...],
    vector_field: str,
) -> ResolvedExactSearch:
    """Validate exact-search-only options and create compiler input."""
    validate_query(query)
    if query.index is not None:
        raise ValueError("index cannot be set when bypassing the vector index")
    if query.probe_count is not None:
        raise ValueError("nprobes cannot be set when bypassing the vector index")
    return ResolvedExactSearch(
        source=query.source,
        source_relation=_freeze_relation(source_relation),
        query_vector=query.query,
        vector_field=vector_field,
        projection=projection,
        predicate=query.predicate,
        limit=query.result_limit,
    )


def resolve_indexed_search(
    query: VectorQuery,
    *,
    index: str,
    metadata: Mapping[str, Any],
    projection: tuple[str, ...],
) -> ResolvedSearch:
    """Resolve portable IVF metadata and query options once for every backend."""
    validate_query(query)
    snapshot = current_snapshot(metadata)
    dimension = positive_parameter(snapshot, "dimension")
    nlist = positive_parameter(snapshot, "nlist")
    ntotal = positive_parameter(snapshot, "ntotal")
    if len(query.query) != dimension:
        raise ValueError(
            f"query vector dimension {len(query.query)} does not match index dimension "
            f"{dimension}"
        )
    nprobe = query.probe_count if query.probe_count is not None else min(nlist, 20)
    if nprobe <= 0 or nprobe > nlist:
        raise ValueError(f"nprobe must be in 1..={nlist}")

    source_key_fields = tuple(str(field) for field in snapshot["source-key-fields"])
    vector_field = str(snapshot["vector-field"])
    posting_encoding = str(snapshot["parameters"]["posting_encoding"])
    if posting_encoding not in {"source", "lvq4", "lvq8"}:
        raise ValueError(f"unsupported IVF posting encoding: {posting_encoding}")
    metric = str(snapshot["metric"])
    query_vector = _transform_query(query.query, metric)
    relations = snapshot["index-relations"]
    needs_source = (
        posting_encoding == "source"
        or query.predicate is not None
        or any(field not in source_key_fields for field in projection)
        or query.projection is None
    )
    return ResolvedSearch(
        source=query.source,
        source_relation=_freeze_relation(snapshot["source"]),
        index=index,
        centroids_relation=_freeze_relation(relations["ivf_centroids"]),
        postings_relation=_freeze_relation(relations["ivf_postings"]),
        query_vector=query_vector,
        projection=projection,
        predicate=query.predicate,
        limit=query.result_limit,
        dimension=dimension,
        nlist=nlist,
        nprobe=nprobe,
        source_key_fields=source_key_fields,
        vector_field=vector_field,
        posting_encoding=posting_encoding,
        needs_source=needs_source,
        snapshot_id=int(snapshot["snapshot-id"]),
        family=str(snapshot["index-family"]),
        index_schema_version=int(snapshot["index-schema-version"]),
        metric=metric,
        ntotal=ntotal,
    )


def current_snapshot(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the current validated snapshot from an index metadata document."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    snapshot_id = metadata["current-snapshot-id"]
    for snapshot in metadata["snapshots"]:
        if snapshot["snapshot-id"] == snapshot_id:
            return snapshot
    raise ValueError("current index snapshot is missing")


def positive_parameter(snapshot: Mapping[str, Any], name: str) -> int:
    value = int(snapshot["parameters"][name])
    if value <= 0:
        raise ValueError(f"invalid IVF parameter: {name}")
    return value


def _transform_query(query: tuple[float, ...], metric: str) -> tuple[float, ...]:
    canonical = tuple(
        struct.unpack("!f", struct.pack("!f", value))[0] for value in query
    )
    if metric == "l2_squared":
        return canonical
    if metric != "cosine":
        raise ValueError(f"unsupported IVF metric: {metric}")
    norm = math.sqrt(sum(value * value for value in canonical))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("cosine query vector must have a finite, non-zero norm")
    return tuple(
        struct.unpack("!f", struct.pack("!f", value / norm))[0] for value in canonical
    )


def _freeze_relation(relation: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(relation, Mapping):
        raise TypeError("relation must be a mapping")
    return _freeze_mapping(relation)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(child) for key, child in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(child) for child in value)
    return value

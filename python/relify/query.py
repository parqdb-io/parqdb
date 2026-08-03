from __future__ import annotations

from dataclasses import dataclass, replace

from .identifier import TableIdentifier


@dataclass(frozen=True, slots=True)
class VectorQuery:
    """Backend-independent vector search intent."""

    source: TableIdentifier
    query: tuple[float, ...]
    column: str | None = None
    index: str | None = None
    projection: tuple[str, ...] | None = None
    result_limit: int = 10
    probe_count: int | None = None
    predicate: str | None = None
    bypass_index: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, TableIdentifier):
            raise TypeError("source must be a relify.TableIdentifier")

    def select(self, columns: list[str]) -> VectorQuery:
        return replace(self, projection=tuple(columns))

    def limit(self, limit: int) -> VectorQuery:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return replace(self, result_limit=limit)

    def nprobes(self, nprobes: int) -> VectorQuery:
        if not isinstance(nprobes, int) or isinstance(nprobes, bool) or nprobes <= 0:
            raise ValueError("nprobes must be a positive integer")
        return replace(self, probe_count=nprobes)

    def where(self, predicate: str) -> VectorQuery:
        if not isinstance(predicate, str):
            raise TypeError("predicate must be a string")
        if not predicate.strip():
            raise ValueError("predicate must not be empty")
        return replace(self, predicate=predicate)

    def bypass_vector_index(self) -> VectorQuery:
        return replace(self, bypass_index=True)

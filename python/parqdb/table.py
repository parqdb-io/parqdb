from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from typing import Protocol, SupportsFloat, runtime_checkable

from .build import IndexStatus
from .config import IVF, WriteOptions
from .identifier import TableIdentifier
from .query import VectorQuery
from .runtime.catalog import IndexInfo


@runtime_checkable
class Table(Protocol):
    """Portable source-table surface."""

    @property
    def identifier(self) -> TableIdentifier: ...

    def create_index(
        self,
        index: str,
        *,
        column: str,
        key: list[str],
        config: IVF,
        writer_options: WriteOptions | None = None,
        wait_timeout: timedelta | None = None,
    ) -> None: ...

    def register_index(
        self,
        index: str,
        *,
        metadata_location: str,
    ) -> None: ...

    def refresh_index(
        self,
        index: str,
        *,
        config: IVF | None = None,
        writer_options: WriteOptions | None = None,
        wait_timeout: timedelta | None = None,
    ) -> None: ...

    def index_status(self, index: str) -> IndexStatus: ...

    def wait_for_index(
        self,
        index: str,
        *,
        timeout: timedelta = timedelta(minutes=5),
    ) -> None: ...

    def list_indexes(self) -> list[IndexInfo]: ...

    def drop_index(self, index: str) -> None: ...

    def search(
        self,
        query: Iterable[SupportsFloat],
        *,
        column: str | None = None,
        index: str | None = None,
    ) -> VectorQuery: ...


def _normalize_query(query: Iterable[SupportsFloat]) -> tuple[float, ...]:
    ndim = getattr(query, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError("query must be a one-dimensional vector")

    tolist = getattr(query, "tolist", None)
    values: object = tolist() if callable(tolist) else query
    if not isinstance(values, Iterable):
        raise ValueError("query must be a one-dimensional iterable of numeric values")
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "query must be a one-dimensional iterable of numeric values"
        ) from error

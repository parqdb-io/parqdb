from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from typing import Any, Protocol, SupportsFloat, runtime_checkable

from .build import IndexStatus
from .builders.v1 import IndexBuilder
from .catalog import IndexInfo
from .config import IVF, WriteOptions
from .identifier import TableIdentifier
from .query import VectorQuery


@runtime_checkable
class Table(Protocol):
    """Portable source-table surface shared by every Relify backend."""

    @property
    def identifier(self) -> TableIdentifier: ...

    def create_index(
        self,
        index: str,
        *,
        column: str,
        key: list[str],
        config: IVF,
        builder: IndexBuilder | None = None,
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


class TableOperations:
    """Reusable index lifecycle and search operations for concrete tables."""

    _session: Any
    _identifier: TableIdentifier

    @property
    def identifier(self) -> TableIdentifier:
        return self._identifier

    def create_index(
        self,
        index: str,
        *,
        column: str,
        key: list[str],
        config: IVF,
        builder: IndexBuilder | None = None,
        writer_options: WriteOptions | None = None,
        wait_timeout: timedelta | None = None,
    ) -> None:
        self._session._builds.create(
            self._identifier,
            index=index,
            column=column,
            key=key,
            config=config,
            builder=builder,
            writer_options=writer_options,
            wait_timeout=wait_timeout,
        )

    def index_status(self, index: str) -> IndexStatus:
        return self._session._builds.status(self._identifier, index)

    def wait_for_index(
        self,
        index: str,
        *,
        timeout: timedelta = timedelta(minutes=5),
    ) -> None:
        self._session._builds.wait(self._identifier, index, timeout)

    def list_indexes(self) -> list[IndexInfo]:
        return self._session._list_table_indexes(self._identifier)

    def drop_index(self, index: str) -> None:
        self._session._drop_table_index(self._identifier, index)

    def search(
        self,
        query: Iterable[SupportsFloat],
        *,
        column: str | None = None,
        index: str | None = None,
    ) -> VectorQuery:
        return VectorQuery(
            source=self._identifier,
            query=_normalize_query(query),
            column=column,
            index=index,
        )


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

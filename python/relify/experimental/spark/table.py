from __future__ import annotations

from typing import TYPE_CHECKING

from ...identifier import TableIdentifier
from ...table import TableOperations

if TYPE_CHECKING:
    from .session import Session


class Table(TableOperations):
    """A source table resolved by a Spark Relify session."""

    def __init__(self, session: Session, identifier: TableIdentifier) -> None:
        self._session = session
        self._identifier = identifier

    def __repr__(self) -> str:
        return f"relify.experimental.spark.Table({self._identifier!r})"

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from .session import Session


@dataclass(frozen=True)
class MaintenanceObject:
    kind: Literal["metadata", "index_data"]
    reference: str
    modified_at: datetime


class Maintenance:
    def __init__(self, session: Session) -> None:
        self._session = session

    def remove_orphans(
        self,
        *,
        older_than: datetime,
        dry_run: bool = True,
    ) -> tuple[MaintenanceObject, ...]:
        if not isinstance(older_than, datetime):
            raise TypeError("older_than must be a datetime")
        if older_than.tzinfo is None or older_than.utcoffset() is None:
            raise ValueError("older_than must be timezone-aware")
        if older_than > datetime.now(UTC):
            raise ValueError("older_than must not be in the future")
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be a boolean")
        older_than_ms = int(older_than.timestamp() * 1000)
        return tuple(
            MaintenanceObject(
                kind=cast(Literal["metadata", "index_data"], kind),
                reference=reference,
                modified_at=datetime.fromtimestamp(
                    modified_ms / 1000,
                    tz=UTC,
                ),
            )
            for kind, reference, modified_ms in self._session._native.remove_orphans(
                older_than_ms,
                dry_run,
            )
        )

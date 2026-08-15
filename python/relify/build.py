from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class IndexStatus:
    state: Literal["pending", "building", "ready", "failed"]
    progress: float | None = None
    phase: str | None = None
    completed: int | None = None
    total: int | None = None
    current_snapshot_id: int | None = None
    error: str | None = None

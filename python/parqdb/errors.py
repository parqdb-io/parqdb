from __future__ import annotations

from ._native import ParqDBError


class UnsupportedOperationError(ParqDBError):
    """Raised when an operation is unavailable for the selected deployment."""


class ServiceUnavailableError(ParqDBError):
    """Raised when a remote ParqDB service cannot be reached."""


class StreamExecutionError(ParqDBError):
    """Raised when a remote query fails after its response has started."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id

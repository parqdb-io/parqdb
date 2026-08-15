from __future__ import annotations

from ._native import RelifyError


class UnsupportedOperationError(RelifyError):
    """Raised when an operation is unavailable for the selected deployment."""


class ServiceUnavailableError(RelifyError):
    """Raised when a remote Relify service cannot be reached."""


class StreamExecutionError(RelifyError):
    """Raised when a remote query fails after its response has started."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id

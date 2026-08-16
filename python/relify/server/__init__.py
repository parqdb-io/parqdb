from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..datafusion import RuntimeEnvBuilder
from ..datafusion import SessionConfig as DataFusionSessionConfig


class ASGIApp(Protocol):
    def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> Awaitable[None]: ...


def create_app(
    root: str | Path,
    *,
    warehouse: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    config: DataFusionSessionConfig | None = None,
    runtime: RuntimeEnvBuilder | None = None,
    allowed_source_prefixes: Sequence[str | Path] = (),
) -> ASGIApp:
    """Create an ASGI server with explicit server-visible source prefixes."""
    try:
        from .app import create_http_app
    except ModuleNotFoundError as error:
        if error.name != "starlette":
            raise
        raise RuntimeError(
            "the Relify server requires the 'server' optional dependency"
        ) from error
    return create_http_app(
        root,
        warehouse=warehouse,
        storage_options=storage_options,
        config=config,
        runtime=runtime,
        allowed_source_prefixes=allowed_source_prefixes,
    )


__all__ = ["ASGIApp", "create_app"]

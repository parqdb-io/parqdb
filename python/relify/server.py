from __future__ import annotations

from collections.abc import Awaitable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .datafusion import RuntimeEnvBuilder
from .datafusion import SessionConfig as DataFusionSessionConfig


class ASGIApp(Protocol):
    def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> Awaitable[None]: ...


def create_app(
    root: str | Path | None = None,
    *,
    catalog: str | None = None,
    index_root: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    config: DataFusionSessionConfig | None = None,
    runtime: RuntimeEnvBuilder | None = None,
) -> ASGIApp:
    """Create the optional ASGI server for one Relify catalog."""
    try:
        from ._http_server import create_http_app
    except ModuleNotFoundError as error:
        if error.name != "starlette":
            raise
        raise RuntimeError(
            "the Relify server requires the 'server' optional dependency"
        ) from error
    return create_http_app(
        root,
        catalog=catalog,
        index_root=index_root,
        storage_options=storage_options,
        config=config,
        runtime=runtime,
    )


__all__ = ["ASGIApp", "create_app"]

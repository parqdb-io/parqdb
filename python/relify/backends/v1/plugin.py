from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import pyarrow

from ...catalog import IndexCatalog
from ...query import VectorQuery
from ...table import Table
from .capabilities import BackendCapabilities, CapabilityReport, Terminal

BACKEND_API_VERSION = 1


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Identity and compatibility information for one backend plugin."""

    name: str
    display_name: str
    distribution: str
    api_version: int = BACKEND_API_VERSION
    documentation: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not self.name.isascii()
            or self.name.lower() != self.name
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in self.name
            )
        ):
            raise ValueError(
                "backend name must contain lowercase ASCII letters, digits, '_' or '-'"
            )
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("backend display_name must not be empty")
        if not isinstance(self.distribution, str) or not self.distribution.strip():
            raise ValueError("backend distribution must not be empty")
        if not isinstance(self.api_version, int) or isinstance(
            self.api_version,
            bool,
        ):
            raise TypeError("backend api_version must be an integer")
        if self.api_version <= 0:
            raise ValueError("backend api_version must be positive")
        if self.documentation is not None and (
            not isinstance(self.documentation, str) or not self.documentation.strip()
        ):
            raise ValueError("backend documentation must not be empty")


@runtime_checkable
class BackendSession(Protocol):
    """Minimum portable surface of any connected Relify backend session."""

    @property
    def backend(self) -> BackendInfo: ...

    @property
    def capabilities(self) -> CapabilityReport: ...

    @property
    def indexes(self) -> IndexCatalog: ...

    def table(self, identifier: str) -> Table: ...


@runtime_checkable
class QueryBackendSession(BackendSession, Protocol):
    """Additional portable surface required when query profiles are declared."""

    def collect(self, query: VectorQuery) -> pyarrow.Table: ...


@runtime_checkable
class BackendPlugin(Protocol):
    """Discoverable factory for one concrete backend session type."""

    @property
    def info(self) -> BackendInfo: ...

    @property
    def declared_capabilities(self) -> BackendCapabilities: ...

    def connect(self, *args: Any, **kwargs: Any) -> BackendSession: ...


@dataclass(frozen=True, slots=True)
class SimpleBackendPlugin:
    """Small plugin implementation suitable for built-in backends."""

    info: BackendInfo
    declared_capabilities: BackendCapabilities
    connector: Callable[..., Any]

    def connect(self, *args: Any, **kwargs: Any) -> BackendSession:
        session = self.connector(*args, **kwargs)
        if not isinstance(session, BackendSession):
            raise TypeError("backend connector returned an invalid session")
        if session.backend != self.info:
            raise TypeError("backend session identity does not match its plugin")
        if session.capabilities.declared != self.declared_capabilities:
            raise TypeError(
                "backend session capabilities do not match its plugin declaration"
            )
        if self.declared_capabilities.query_profiles and not isinstance(
            session,
            QueryBackendSession,
        ):
            raise TypeError(
                "backend declares query profiles but returned a non-query session"
            )
        _validate_terminal_methods(session)
        return cast(BackendSession, session)


_TERMINAL_METHODS = {
    Terminal.COLLECT: "collect",
    Terminal.DATAFRAME: "to_dataframe",
    Terminal.RELATION: "to_relation",
    Terminal.SQL: "to_sql",
    Terminal.EXPLAIN: "explain",
    Terminal.ANALYZE: "analyze",
}


def _validate_terminal_methods(session: BackendSession) -> None:
    for terminal in session.capabilities.available.terminals:
        method = _TERMINAL_METHODS[terminal]
        if not callable(getattr(session, method, None)):
            raise TypeError(
                f"backend reports terminal {terminal.value!r} but the session "
                f"does not implement {method}()"
            )

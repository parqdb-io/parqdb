from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import relify
from relify.backends import _registry
from relify.backends.v1 import (
    BackendCapabilities,
    BackendInfo,
    CapabilityReport,
    CapabilityStatus,
    QueryProfile,
    SimpleBackendPlugin,
    Terminal,
)


@pytest.fixture(autouse=True)
def clear_backend_registry() -> None:
    _registry._clear_for_tests()
    yield
    _registry._clear_for_tests()


def test_builtin_backends_are_discoverable_without_loading() -> None:
    installed = relify.backends.installed()

    assert [backend.name for backend in installed] == ["local"]
    assert all(backend.distribution == "relify" for backend in installed)
    assert all(backend.built_in for backend in installed)


def test_builtin_plugin_connects_a_concrete_session(tmp_path: Path) -> None:
    plugin = relify.backends.load("local")

    session = plugin.connect(tmp_path / "relify-data")

    assert session.backend == plugin.info
    assert session.capabilities.declared == plugin.declared_capabilities
    assert isinstance(session, relify.Session)


def test_bound_capabilities_distinguish_unavailable_from_unsupported(
    tmp_path: Path,
) -> None:
    session = relify.connect(tmp_path / "relify-data")
    parquet = QueryProfile("ivf", "parquet", "parquet")
    iceberg = QueryProfile("ivf", "iceberg", "iceberg")
    unknown = QueryProfile("future", "parquet", "parquet")

    assert session.capabilities.status(parquet) is CapabilityStatus.SUPPORTED
    assert session.capabilities.status(iceberg) is CapabilityStatus.UNAVAILABLE
    assert session.capabilities.reason(iceberg) == (
        "the session has no Iceberg catalog"
    )
    assert session.capabilities.status(unknown) is CapabilityStatus.UNSUPPORTED
    inventory = session.capabilities.to_dict()
    available = inventory["available"]
    assert isinstance(available, dict)
    assert available["terminals"] == [
        "analyze",
        "collect",
        "dataframe",
        "explain",
        "sql",
    ]
    assert inventory["unavailable"] == [
        {
            "capability": {
                "kind": "query_profile",
                "family": "ivf",
                "source_profile": "iceberg",
                "index_profile": "iceberg",
            },
            "reason": "the session has no Iceberg catalog",
        }
    ]


def test_capability_report_rejects_an_available_undeclared_capability() -> None:
    declared = BackendCapabilities(
        query_profiles=frozenset({QueryProfile("ivf", "parquet", "parquet")}),
        terminals=frozenset({Terminal.COLLECT}),
    )
    available = BackendCapabilities(
        query_profiles=frozenset({QueryProfile("ivf", "iceberg", "iceberg")}),
        terminals=frozenset({Terminal.COLLECT}),
    )

    with pytest.raises(ValueError, match="must be declared"):
        CapabilityReport(declared, available)


def test_query_profile_requires_portable_collect_terminal() -> None:
    with pytest.raises(ValueError, match="require the collect terminal"):
        BackendCapabilities(
            query_profiles=frozenset({QueryProfile("ivf", "parquet", "parquet")})
        )


def test_plugin_rejects_a_missing_declared_terminal() -> None:
    info = BackendInfo("invalid", "Invalid", "relify-invalid")
    capabilities = BackendCapabilities(terminals=frozenset({Terminal.EXPLAIN}))
    session = _FakeSession(info, capabilities)
    plugin = SimpleBackendPlugin(info, capabilities, lambda: session)

    with pytest.raises(TypeError, match="does not implement explain"):
        plugin.connect()


def test_in_process_backend_registration_supports_dynamic_applications() -> None:
    plugin, session = _plugin("private")

    relify.backends.register(plugin)

    assert relify.backends.load("private") is plugin
    assert relify.backends.connect("private", object()) is session
    assert [backend.name for backend in relify.backends.installed()] == [
        "local",
        "private",
    ]


def test_experimental_backends_are_not_registered_as_stable_builtins() -> None:
    assert not hasattr(relify, "spark")
    assert not hasattr(relify, "starrocks")
    assert not hasattr(relify, "Spark")
    assert relify.experimental.Spark is relify.experimental.spark.Spark

    for name in ("spark", "starrocks"):
        with pytest.raises(relify.backends.BackendNotFoundError):
            relify.backends.load(name)


def test_external_backend_entry_point_is_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, _ = _plugin("duckdb")
    endpoint = _EntryPoint("duckdb", plugin)
    monkeypatch.setattr(
        _registry,
        "entry_points",
        lambda *, group: [endpoint] if group == "relify.backends" else [],
    )

    installed = relify.backends.installed()

    assert endpoint.load_count == 0
    duckdb = next(backend for backend in installed if backend.name == "duckdb")
    assert duckdb.distribution == "relify-duckdb"
    assert duckdb.version == "1.2.3"

    assert relify.backends.load("duckdb") is plugin
    assert endpoint.load_count == 1
    assert relify.backends.load("duckdb") is plugin
    assert endpoint.load_count == 1


def test_backend_loader_rejects_an_incompatible_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = SimpleBackendPlugin(
        BackendInfo(
            "future",
            "Future",
            "relify-future",
            api_version=2,
        ),
        BackendCapabilities(),
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        _registry,
        "entry_points",
        lambda *, group: (
            [_EntryPoint("future", plugin)] if group == "relify.backends" else []
        ),
    )

    with pytest.raises(
        relify.backends.IncompatibleBackendError,
        match="uses API version 2",
    ):
        relify.backends.load("future")


def test_unknown_backend_has_an_actionable_error() -> None:
    with pytest.raises(
        relify.backends.BackendNotFoundError,
        match="install its Relify integration package",
    ):
        relify.backends.load("missing")


def test_backend_import_failure_names_the_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _EntryPoint("broken", None)
    endpoint.error = ImportError("missing driver")
    monkeypatch.setattr(
        _registry,
        "entry_points",
        lambda *, group: [endpoint] if group == "relify.backends" else [],
    )

    with pytest.raises(
        relify.backends.BackendLoadError,
        match=r"relify_duckdb\.backend:plugin",
    ):
        relify.backends.load("broken")


def _plugin(name: str) -> tuple[SimpleBackendPlugin, _FakeSession]:
    info = BackendInfo(name, name.title(), f"relify-{name}")
    capabilities = BackendCapabilities(terminals=frozenset({Terminal.COLLECT}))
    session = _FakeSession(info, capabilities)
    return (
        SimpleBackendPlugin(
            info,
            capabilities,
            lambda *_args, **_kwargs: session,
        ),
        session,
    )


class _FakeSession:
    indexes = object()

    def __init__(
        self,
        backend: BackendInfo,
        capabilities: BackendCapabilities,
    ) -> None:
        self.backend = backend
        self.capabilities = CapabilityReport.fully_available(capabilities)

    def table(self, identifier: Any) -> object:
        return object()

    def collect(self, query: Any) -> list[Any]:
        return []


@dataclass
class _Distribution:
    metadata: dict[str, str]
    version: str


class _EntryPoint:
    module = "relify_duckdb.backend"
    value = "relify_duckdb.backend:plugin"

    def __init__(
        self,
        name: str,
        plugin: SimpleBackendPlugin | None,
    ) -> None:
        self.name = name
        self.dist = _Distribution({"Name": f"relify-{name}"}, "1.2.3")
        self._plugin = plugin
        self.load_count = 0
        self.error: Exception | None = None

    def load(self) -> SimpleBackendPlugin:
        self.load_count += 1
        if self.error is not None:
            raise self.error
        assert self._plugin is not None
        return self._plugin

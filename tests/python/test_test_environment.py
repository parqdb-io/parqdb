from __future__ import annotations

from pathlib import Path

import pytest
from support import capabilities
from support.capabilities import (
    CapabilityProbeError,
    CapabilityRegistry,
    CapabilityState,
)
from support.config import (
    HdfsConfig,
    load_test_environment,
)
from support.config import (
    TestEnvironment as RelifyTestEnvironment,
)
from support.config import (
    TestEnvironmentError as EnvironmentError,
)


def test_environment_expands_secrets_and_parses_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_ACCESS_KEY", "access")
    monkeypatch.setenv("TEST_SECRET_KEY", "secret")
    monkeypatch.setenv(
        "TEST_ICEBERG_PROPERTIES",
        '{"type":"rest","uri":"http://iceberg:8181"}',
    )
    path = tmp_path / "test-env.toml"
    path.write_text(
        """
[s3]
uri = "s3://bucket/prefix"
endpoint = "http://127.0.0.1:9000"
access_key = "${TEST_ACCESS_KEY}"
secret_key = "${TEST_SECRET_KEY}"

[iceberg]
name = "lakehouse"
properties_json = "${TEST_ICEBERG_PROPERTIES}"

[starrocks]
flight_uri = "grpc://127.0.0.1:9408"
catalog_name = "lakehouse"
""",
        encoding="utf-8",
    )

    environment = load_test_environment(path)

    assert environment.s3 is not None
    assert environment.s3.access_key == "access"
    assert environment.s3.secret_key == "secret"
    assert environment.iceberg is not None
    assert environment.iceberg.properties["type"] == "rest"
    assert environment.starrocks is not None
    assert environment.starrocks.db_kwargs == {}


def test_environment_rejects_an_unset_placeholder(tmp_path: Path) -> None:
    path = tmp_path / "test-env.toml"
    path.write_text(
        """
[s3]
uri = "s3://bucket"
endpoint = "http://127.0.0.1:9000"
access_key = "${RELIFY_TEST_UNSET_ACCESS_KEY}"
secret_key = "secret"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        EnvironmentError,
        match="RELIFY_TEST_UNSET_ACCESS_KEY",
    ):
        load_test_environment(path)


def test_environment_rejects_an_incomplete_starrocks_catalog(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test-env.toml"
    path.write_text(
        """
[starrocks]
flight_uri = "grpc://127.0.0.1:9408"
catalog_name = "lakehouse"
""",
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentError, match=r"requires.*\[iceberg\]"):
        load_test_environment(path)


def test_environment_requires_matching_catalog_names(tmp_path: Path) -> None:
    path = tmp_path / "test-env.toml"
    path.write_text(
        """
[iceberg]
name = "lakehouse"
properties_json = "{}"

[starrocks]
flight_uri = "grpc://127.0.0.1:9408"
catalog_name = "other"
""",
        encoding="utf-8",
    )

    with pytest.raises(EnvironmentError, match="must match"):
        load_test_environment(path)


def test_capability_registry_distinguishes_missing_available_and_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = RelifyTestEnvironment(
        path=None,
        hdfs=HdfsConfig(uri=None, mode="managed"),
    )
    registry = CapabilityRegistry(environment)
    assert registry.probe("file").state == CapabilityState.AVAILABLE
    assert registry.probe("s3").state == CapabilityState.MISSING

    def fail_probe(environment: RelifyTestEnvironment) -> None:
        del environment
        raise RuntimeError("unreachable")

    monkeypatch.setitem(capabilities._PROBES, "hdfs", fail_probe)
    with pytest.raises(CapabilityProbeError, match="unreachable"):
        registry.probe("hdfs")
    assert registry.inspect("hdfs").state == CapabilityState.FAILED


def test_capability_registry_rejects_unknown_names() -> None:
    registry = CapabilityRegistry(RelifyTestEnvironment(path=None))

    with pytest.raises(ValueError, match="unknown test capability"):
        registry.configured("unknown")

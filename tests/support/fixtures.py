from __future__ import annotations

from collections.abc import Iterator

import pytest

from support.capabilities import CapabilityProbeError, CapabilityRegistry
from support.config import (
    HdfsConfig,
    IcebergConfig,
    S3Config,
    SparkConfig,
    StarRocksConfig,
    TestEnvironment,
    load_test_environment,
)


@pytest.fixture(scope="session")
def test_env(pytestconfig: pytest.Config) -> TestEnvironment:
    return load_test_environment(pytestconfig.getoption("test_env"))


@pytest.fixture(scope="session")
def capability_registry(test_env: TestEnvironment) -> CapabilityRegistry:
    return CapabilityRegistry(test_env)


@pytest.fixture(autouse=True)
def _probe_required_capabilities(
    request: pytest.FixtureRequest,
    capability_registry: CapabilityRegistry,
) -> Iterator[None]:
    required: set[str] = set()
    for marker in request.node.iter_markers("requires"):
        required.update(str(name) for name in marker.args)
    for name in sorted(required):
        try:
            result = capability_registry.probe(name)
        except CapabilityProbeError as error:
            pytest.fail(str(error), pytrace=False)
        if result.state.value == "missing":
            pytest.fail(
                f"required capability {name!r} is not configured",
                pytrace=False,
            )
    yield


@pytest.fixture(scope="session")
def s3(test_env: TestEnvironment) -> S3Config:
    assert test_env.s3 is not None
    return test_env.s3


@pytest.fixture(scope="session")
def hdfs(test_env: TestEnvironment) -> HdfsConfig:
    assert test_env.hdfs is not None
    return test_env.hdfs


@pytest.fixture(scope="session")
def iceberg(test_env: TestEnvironment) -> IcebergConfig:
    assert test_env.iceberg is not None
    return test_env.iceberg


@pytest.fixture(scope="session")
def spark(test_env: TestEnvironment) -> SparkConfig:
    assert test_env.spark is not None
    return test_env.spark


@pytest.fixture(scope="session")
def starrocks(test_env: TestEnvironment) -> StarRocksConfig:
    assert test_env.starrocks is not None
    return test_env.starrocks

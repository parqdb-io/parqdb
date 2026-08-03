from __future__ import annotations

import shutil
import socket
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit

from support.config import TestEnvironment

CAPABILITY_NAMES = ("file", "s3", "hdfs", "iceberg", "spark", "starrocks")
_DEPENDENCIES = {"starrocks": ("iceberg",)}


class CapabilityProbeError(RuntimeError):
    """Raised when a configured capability cannot be used."""


class CapabilityState(StrEnum):
    AVAILABLE = "available"
    FAILED = "failed"
    MISSING = "missing"


@dataclass(frozen=True)
class CapabilityResult:
    name: str
    state: CapabilityState
    detail: str


class CapabilityRegistry:
    def __init__(self, environment: TestEnvironment) -> None:
        self.environment = environment
        self._results: dict[str, CapabilityResult] = {}

    def configured(self, name: str) -> bool:
        self._validate_name(name)
        return self.environment.configured(name)

    def probe(self, name: str) -> CapabilityResult:
        self._validate_name(name)
        cached = self._results.get(name)
        if cached is not None:
            if cached.state == CapabilityState.FAILED:
                raise CapabilityProbeError(cached.detail)
            return cached
        if not self.configured(name):
            result = CapabilityResult(
                name,
                CapabilityState.MISSING,
                "not configured",
            )
            self._results[name] = result
            return result
        try:
            for dependency in _DEPENDENCIES.get(name, ()):
                dependency_result = self.probe(dependency)
                if dependency_result.state != CapabilityState.AVAILABLE:
                    raise CapabilityProbeError(
                        f"required capability {dependency} is not available"
                    )
            _PROBES[name](self.environment)
        except Exception as error:
            detail = f"{name} probe failed: {error}"
            self._results[name] = CapabilityResult(
                name,
                CapabilityState.FAILED,
                detail,
            )
            raise CapabilityProbeError(detail) from error
        result = CapabilityResult(
            name,
            CapabilityState.AVAILABLE,
            "probe succeeded",
        )
        self._results[name] = result
        return result

    def inspect(self, name: str) -> CapabilityResult:
        try:
            return self.probe(name)
        except CapabilityProbeError:
            return self._results[name]

    @staticmethod
    def _validate_name(name: str) -> None:
        if name not in CAPABILITY_NAMES:
            supported = ", ".join(CAPABILITY_NAMES)
            raise ValueError(
                f"unknown test capability {name!r}; expected one of: {supported}"
            )


def _probe_file(environment: TestEnvironment) -> None:
    del environment


def _probe_s3(environment: TestEnvironment) -> None:
    import pyarrow.fs as fs

    config = environment.s3
    assert config is not None
    endpoint = urlsplit(config.endpoint)
    endpoint_override = endpoint.hostname
    if endpoint.port is not None:
        endpoint_override = f"{endpoint_override}:{endpoint.port}"
    filesystem = fs.S3FileSystem(
        access_key=config.access_key,
        secret_key=config.secret_key,
        endpoint_override=endpoint_override,
        scheme=endpoint.scheme,
        region=config.region,
        allow_bucket_creation=True,
    )
    location = urlsplit(config.uri)
    path = location.netloc
    if location.path.strip("/"):
        path = f"{path}/{location.path.strip('/')}"
    filesystem.get_file_info(path)


def _probe_hdfs(environment: TestEnvironment) -> None:
    config = environment.hdfs
    assert config is not None
    if config.mode == "managed":
        missing = [
            command
            for command in ("java", "mvn", "kdestroy")
            if shutil.which(command) is None
        ]
        if missing:
            raise RuntimeError("managed MiniDFS requires: " + ", ".join(missing))
        return
    assert config.uri is not None
    uri = urlsplit(config.uri)
    if uri.hostname is None or uri.port is None:
        raise RuntimeError("external HDFS URI must include host and port")
    with socket.create_connection((uri.hostname, uri.port), timeout=3):
        pass


def _probe_iceberg(environment: TestEnvironment) -> None:
    from pyiceberg.catalog import load_catalog

    config = environment.iceberg
    assert config is not None
    catalog = load_catalog(config.name, **config.properties)
    catalog.list_namespaces()


def _probe_spark(environment: TestEnvironment) -> None:
    del environment
    if shutil.which("java") is None:
        raise RuntimeError("Spark requires Java on PATH")
    import pyspark

    if not pyspark.__version__:
        raise RuntimeError("PySpark did not report a version")


def _probe_starrocks(environment: TestEnvironment) -> None:
    from adbc_driver_flightsql import dbapi

    config = environment.starrocks
    assert config is not None
    connection = dbapi.connect(
        uri=config.flight_uri,
        db_kwargs=config.db_kwargs,
    )
    try:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        finally:
            cursor.close()
    finally:
        connection.close()


_PROBES: dict[str, Callable[[TestEnvironment], None]] = {
    "file": _probe_file,
    "s3": _probe_s3,
    "hdfs": _probe_hdfs,
    "iceberg": _probe_iceberg,
    "spark": _probe_spark,
    "starrocks": _probe_starrocks,
}

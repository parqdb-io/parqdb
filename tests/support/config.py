from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class TestEnvironmentError(ValueError):
    """Raised when a configured test environment is incomplete or invalid."""


@dataclass(frozen=True)
class S3Config:
    uri: str
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"

    @property
    def storage_options(self) -> dict[str, str]:
        return {
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
            "aws_region": self.region,
            "aws_endpoint": self.endpoint,
            "aws_allow_http": str(self.endpoint.startswith("http://")).lower(),
            "aws_virtual_hosted_style_request": "false",
        }


@dataclass(frozen=True)
class HdfsConfig:
    uri: str | None
    mode: str


@dataclass(frozen=True)
class IcebergConfig:
    name: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class SparkConfig:
    iceberg_version: str


@dataclass(frozen=True)
class StarRocksConfig:
    flight_uri: str
    catalog_name: str
    db_kwargs: dict[str, Any]


@dataclass(frozen=True)
class TestEnvironment:
    path: Path | None
    s3: S3Config | None = None
    hdfs: HdfsConfig | None = None
    iceberg: IcebergConfig | None = None
    spark: SparkConfig | None = None
    starrocks: StarRocksConfig | None = None

    def configured(self, capability: str) -> bool:
        if capability == "file":
            return True
        return getattr(self, capability, None) is not None


def load_test_environment(path: str | Path | None) -> TestEnvironment:
    resolved = _resolve_path(path)
    if resolved is None:
        return TestEnvironment(path=None)

    try:
        document = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TestEnvironmentError(
            f"cannot read test environment {resolved}: {error}"
        ) from error
    expanded = _expand_environment(document, location=str(resolved))

    s3 = _parse_s3(expanded.get("s3"))
    hdfs = _parse_hdfs(expanded.get("hdfs"))
    iceberg = _parse_iceberg(expanded.get("iceberg"))
    spark = _parse_spark(expanded.get("spark"))
    starrocks = _parse_starrocks(expanded.get("starrocks"))
    if starrocks is not None:
        if iceberg is None:
            raise TestEnvironmentError(
                "[starrocks] requires a matching [iceberg] section"
            )
        if starrocks.catalog_name != iceberg.name:
            raise TestEnvironmentError(
                "[starrocks].catalog_name must match [iceberg].name"
            )
    return TestEnvironment(
        path=resolved,
        s3=s3,
        hdfs=hdfs,
        iceberg=iceberg,
        spark=spark,
        starrocks=starrocks,
    )


def default_test_environment_path() -> Path:
    return Path(__file__).parents[1] / "test-env.toml"


def _resolve_path(path: str | Path | None) -> Path | None:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise TestEnvironmentError(
                f"test environment file does not exist: {resolved}"
            )
        return resolved
    default = default_test_environment_path()
    return default.resolve() if default.is_file() else None


def _expand_environment(value: Any, *, location: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _expand_environment(
                nested,
                location=f"{location}:{key}",
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_expand_environment(nested, location=location) for nested in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise TestEnvironmentError(
                f"{location} references unset environment variable {name}"
            )
        return os.environ[name]

    return _ENVIRONMENT_VARIABLE.sub(replace, value)


def _parse_s3(section: Any) -> S3Config | None:
    values = _section(section, "s3")
    if values is None:
        return None
    uri = _required_string(values, "uri", "s3")
    _validate_uri(uri, "s3", "s3.uri")
    endpoint = _required_string(values, "endpoint", "s3")
    endpoint_uri = urlsplit(endpoint)
    if endpoint_uri.scheme not in {"http", "https"} or not endpoint_uri.hostname:
        raise TestEnvironmentError(
            "[s3].endpoint must be an absolute http:// or https:// URI"
        )
    return S3Config(
        uri=uri.rstrip("/"),
        endpoint=endpoint.rstrip("/"),
        access_key=_required_string(values, "access_key", "s3"),
        secret_key=_required_string(values, "secret_key", "s3"),
        region=_optional_string(values, "region", "s3", "us-east-1"),
    )


def _parse_hdfs(section: Any) -> HdfsConfig | None:
    values = _section(section, "hdfs")
    if values is None:
        return None
    mode = _optional_string(values, "mode", "hdfs", "external")
    if mode not in {"external", "managed"}:
        raise TestEnvironmentError("[hdfs].mode must be either 'external' or 'managed'")
    uri = values.get("uri")
    if uri is not None and not isinstance(uri, str):
        raise TestEnvironmentError("[hdfs].uri must be a string")
    if mode == "external":
        uri = _required_string(values, "uri", "hdfs")
        _validate_uri(uri, "hdfs", "hdfs.uri")
    return HdfsConfig(uri=uri.rstrip("/") if uri else None, mode=mode)


def _parse_iceberg(section: Any) -> IcebergConfig | None:
    values = _section(section, "iceberg")
    if values is None:
        return None
    return IcebergConfig(
        name=_required_string(values, "name", "iceberg"),
        properties=_json_object(values, "properties_json", "iceberg"),
    )


def _parse_spark(section: Any) -> SparkConfig | None:
    values = _section(section, "spark")
    if values is None:
        return None
    return SparkConfig(
        iceberg_version=_optional_string(
            values,
            "iceberg_version",
            "spark",
            "1.11.0",
        )
    )


def _parse_starrocks(section: Any) -> StarRocksConfig | None:
    values = _section(section, "starrocks")
    if values is None:
        return None
    flight_uri = _required_string(values, "flight_uri", "starrocks")
    _validate_uri(flight_uri, "grpc", "starrocks.flight_uri")
    return StarRocksConfig(
        flight_uri=flight_uri,
        catalog_name=_required_string(values, "catalog_name", "starrocks"),
        db_kwargs=_json_object(
            values,
            "db_kwargs_json",
            "starrocks",
            default={},
        ),
    )


def _section(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TestEnvironmentError(f"[{name}] must be a TOML table")
    return value


def _required_string(values: dict[str, Any], key: str, section: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise TestEnvironmentError(f"[{section}].{key} must be a non-empty string")
    return value


def _optional_string(
    values: dict[str, Any],
    key: str,
    section: str,
    default: str,
) -> str:
    value = values.get(key, default)
    if not isinstance(value, str) or not value:
        raise TestEnvironmentError(f"[{section}].{key} must be a non-empty string")
    return value


def _json_object(
    values: dict[str, Any],
    key: str,
    section: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = values.get(key)
    if value is None and default is not None:
        return default
    if not isinstance(value, str):
        raise TestEnvironmentError(f"[{section}].{key} must be a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise TestEnvironmentError(
            f"[{section}].{key} is not valid JSON: {error.msg}"
        ) from error
    if not isinstance(parsed, dict):
        raise TestEnvironmentError(f"[{section}].{key} must contain a JSON object")
    return parsed


def _validate_uri(value: str, scheme: str, field: str) -> None:
    uri = urlsplit(value)
    if uri.scheme != scheme or not uri.netloc:
        raise TestEnvironmentError(f"[{field}] must be an absolute {scheme}:// URI")

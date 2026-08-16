"""Deployment configuration for a Relify server."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from ..config import SessionConfig

DEFAULT_CONFIG_FILENAME = "relify.toml"

DEFAULT_CONFIG_TEMPLATE = """# Relify server configuration.
#
# Run `relify serve` in this directory, or pass this file with --config.

[server]
# Persistent catalog and index state. Relative paths are resolved from this file.
root = "./relify"
host = "127.0.0.1"
port = 8000

# A client may register only sources under one of these server-visible prefixes.
# Keep this list empty until the server should expose a source location.
allowed_source_prefixes = []

# To store index relations in a separate warehouse, set:
# warehouse = "s3://bucket/relify/"

# Process-wide object storage options. Keep credentials outside this file.
[storage]
# aws_region = "us-east-1"
# aws_endpoint = "https://s3.example.com"

# Relify and DataFusion session options. Values are strings because they map
# directly to the engine configuration keys.
[session]
# "relify.execution.query_dop" = "8"
# "relify.execution.query_concurrency" = "16"
"""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Validated process configuration loaded from one TOML document."""

    root: Path
    host: str
    port: int
    warehouse: str | None
    allowed_source_prefixes: tuple[str, ...]
    storage_options: Mapping[str, str]
    session_options: Mapping[str, str]

    @classmethod
    def defaults(cls, directory: Path) -> ServerConfig:
        return cls(
            root=(directory / "relify").resolve(),
            host="127.0.0.1",
            port=8000,
            warehouse=None,
            allowed_source_prefixes=(),
            storage_options=MappingProxyType({}),
            session_options=MappingProxyType({}),
        )

    def session_config(self) -> SessionConfig | None:
        if not self.session_options:
            return None
        return SessionConfig(dict(self.session_options))


def load_server_config(path: str | Path) -> ServerConfig:
    """Load one Relify server TOML file with strict shape validation."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"cannot read server configuration {config_path}: {error}"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(
            f"invalid server configuration {config_path}: {error}"
        ) from error
    return _from_document(raw, config_path.parent)


def write_default_server_config(path: str | Path) -> Path:
    """Materialize the default configuration without overwriting a file."""
    config_path = Path(path).expanduser()
    if config_path.exists():
        raise FileExistsError(f"configuration already exists: {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return config_path.resolve()


def _from_document(raw: object, directory: Path) -> ServerConfig:
    document = _table(raw, "configuration")
    _only_keys(document, {"server", "storage", "session"}, "configuration")
    server = _table(document.get("server", {}), "[server]")
    _only_keys(
        server,
        {
            "root",
            "host",
            "port",
            "warehouse",
            "allowed_source_prefixes",
        },
        "[server]",
    )
    defaults = ServerConfig.defaults(directory)
    host = _string(server.get("host", defaults.host), "server.host")
    if not host or any(character.isspace() for character in host):
        raise ValueError("server.host must be a non-empty host name or address")
    port = _port(server.get("port", defaults.port))
    root = _local_path(server.get("root", "./relify"), "server.root", directory)
    warehouse = _optional_string(server.get("warehouse"), "server.warehouse")
    prefixes = _source_prefixes(server.get("allowed_source_prefixes", []), directory)
    storage = _string_table(document.get("storage", {}), "[storage]")
    session = _string_table(document.get("session", {}), "[session]")
    return ServerConfig(
        root=root,
        host=host,
        port=port,
        warehouse=warehouse,
        allowed_source_prefixes=prefixes,
        storage_options=MappingProxyType(storage),
        session_options=MappingProxyType(session),
    )


def _table(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _only_keys(value: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unsupported key: {unknown[0]}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _port(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError("server.port must be an integer from 1 through 65535")
    return value


def _local_path(value: object, name: str, directory: Path) -> Path:
    path = Path(_string(value, name)).expanduser()
    if not path.is_absolute():
        path = directory / path
    return path.resolve()


def _source_prefixes(value: object, directory: Path) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("server.allowed_source_prefixes must be an array of strings")
    return tuple(_source_prefix(item, directory) for item in value)


def _source_prefix(value: object, directory: Path) -> str:
    source = _string(value, "server.allowed_source_prefixes entry")
    parsed = urlsplit(source)
    if parsed.scheme and parsed.scheme != "file":
        return source
    if parsed.scheme == "file":
        return source
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = directory / path
    return str(path.resolve())


def _string_table(value: object, name: str) -> dict[str, str]:
    table = _table(value, name)
    result: dict[str, str] = {}
    for key, item in table.items():
        result[key] = _string(item, f"{name}.{key}")
    return result

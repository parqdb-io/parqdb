"""Command-line entry point for Relify deployments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .server import create_app
from .server.config import (
    DEFAULT_CONFIG_FILENAME,
    ServerConfig,
    load_server_config,
    write_default_server_config,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Relify command-line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "serve":
            return _serve(arguments.config)
        if arguments.command == "config" and arguments.config_command == "init":
            return _config_init(arguments.path)
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    raise AssertionError("argparse accepted an unsupported command")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relify")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run a Relify HTTP server")
    serve.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help=f"TOML configuration file (default: ./{DEFAULT_CONFIG_FILENAME} when present)",
    )

    config = commands.add_parser("config", help="manage server configuration")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    init = config_commands.add_parser(
        "init", help="write the default server configuration"
    )
    init.add_argument(
        "--path",
        type=Path,
        default=Path(DEFAULT_CONFIG_FILENAME),
        metavar="PATH",
        help=f"configuration destination (default: ./{DEFAULT_CONFIG_FILENAME})",
    )
    return parser


def _config_init(path: Path) -> int:
    destination = write_default_server_config(path)
    print(f"wrote {destination}")
    return 0


def _serve(config_path: Path | None) -> int:
    configuration = _configuration_for_serve(config_path)
    import uvicorn

    app = create_app(
        configuration.root,
        warehouse=configuration.warehouse,
        storage_options=configuration.storage_options,
        config=configuration.session_config(),
        allowed_source_prefixes=configuration.allowed_source_prefixes,
    )
    uvicorn.run(app, host=configuration.host, port=configuration.port, workers=1)
    return 0


def _configuration_for_serve(config_path: Path | None) -> ServerConfig:
    if config_path is not None:
        return load_server_config(config_path)
    default_path = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if default_path.exists():
        return load_server_config(default_path)
    print(
        f"{default_path} does not exist; using built-in defaults. "
        "Run 'relify config init' to create it.",
        file=sys.stderr,
    )
    return ServerConfig.defaults(Path.cwd())

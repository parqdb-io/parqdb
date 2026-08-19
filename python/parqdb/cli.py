"""Command-line entry point for ParqDB deployments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .publish import build_index, parse_asset, publish
from .server import create_app
from .server.config import (
    DEFAULT_CONFIG_FILENAME,
    ServerConfig,
    load_server_config,
    write_default_server_config,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ParqDB command-line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "serve":
            return _serve(arguments.config)
        if arguments.command == "config" and arguments.config_command == "init":
            return _config_init(arguments.path)
        if arguments.command == "publish":
            return _publish(arguments)
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    raise AssertionError("argparse accepted an unsupported command")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parqdb")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run a ParqDB HTTP server")
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

    publish_command = commands.add_parser(
        "publish",
        help="build and publish a browser-queryable source table and static index",
    )
    publish_command.add_argument("--source", type=Path, required=True, metavar="PARQUET")
    publish_command.add_argument("--key", required=True, metavar="COLUMN")
    publish_command.add_argument("--destination", required=True, metavar="PATH_OR_S3_URI")
    publish_command.add_argument("--public-url", metavar="HTTPS_URL")
    publish_command.add_argument("--index-manifest", type=Path, metavar="PATH")
    publish_command.add_argument("--vector-column", metavar="COLUMN")
    publish_command.add_argument(
        "--text-column",
        action="append",
        default=[],
        metavar="COLUMN",
        help="embed this string column with pinned MiniLM; may be repeated",
    )
    publish_command.add_argument("--nlist", type=int)
    publish_command.add_argument(
        "--encoding", choices=("lvq4", "lvq8"), default="lvq8"
    )
    publish_command.add_argument(
        "--metric", choices=("cosine", "l2_squared"), default="cosine"
    )
    publish_command.add_argument("--threads", type=int, default=8)
    publish_command.add_argument("--embedding-batch-size", type=int, default=128)
    publish_command.add_argument(
        "--work", type=Path, default=Path(".parqdb-publish"), metavar="PATH"
    )
    publish_command.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="REMOTE_PATH=LOCAL_PATH",
    )
    publish_command.add_argument("--s3-endpoint", metavar="HTTPS_URL")
    publish_command.add_argument("--s3-region", metavar="REGION")
    publish_command.add_argument(
        "--cors-origin", default="https://example.invalid", metavar="ORIGIN"
    )
    publish_command.add_argument("--no-verify-http", action="store_true")
    return parser


def _publish(arguments: argparse.Namespace) -> int:
    assets = tuple(parse_asset(value) for value in arguments.asset)
    index_manifest = arguments.index_manifest
    if index_manifest is None:
        if arguments.nlist is None or arguments.nlist <= 0:
            raise ValueError("--nlist must be positive when building an index")
        built = build_index(
            source=arguments.source,
            source_key=arguments.key,
            work=arguments.work,
            nlist=arguments.nlist,
            encoding=arguments.encoding,
            metric=arguments.metric,
            threads=arguments.threads,
            vector_column=arguments.vector_column,
            text_columns=tuple(arguments.text_column),
            embedding_batch_size=arguments.embedding_batch_size,
        )
        index_manifest = built.manifest
        assets = (*assets, *built.model_assets)
    elif arguments.vector_column is not None or arguments.text_column:
        raise ValueError(
            "--index-manifest cannot be combined with --vector-column or --text-column"
        )
    result = publish(
        index_manifest=index_manifest,
        source=arguments.source,
        source_key=arguments.key,
        destination=arguments.destination,
        public_url=arguments.public_url,
        assets=assets,
        s3_endpoint=arguments.s3_endpoint,
        s3_region=arguments.s3_region,
        verify_http=not arguments.no_verify_http,
        cors_origin=arguments.cors_origin,
    )
    print(
        json.dumps(
            {
                "destination": result.destination,
                "manifest_url": result.manifest_url,
                "files": result.files,
                "bytes": result.bytes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
        "Run 'parqdb config init' to create it.",
        file=sys.stderr,
    )
    return ServerConfig.defaults(Path.cwd())

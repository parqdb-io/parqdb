from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from parqdb.cli import _configuration_for_serve, main
from parqdb.server.config import (
    DEFAULT_CONFIG_TEMPLATE,
    load_server_config,
    write_default_server_config,
)


def test_default_server_config_is_safe_and_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    path = write_default_server_config(tmp_path / "parqdb.toml")

    configuration = load_server_config(path)

    assert configuration.root == (tmp_path / "parqdb").resolve()
    assert configuration.host == "127.0.0.1"
    assert configuration.port == 8000
    assert configuration.allowed_source_prefixes == ()
    assert configuration.storage_options == {}
    assert configuration.session_config() is None


def test_server_config_resolves_source_roots_and_engine_options(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text(
        """
[server]
root = "state"
host = "0.0.0.0"
port = 9042
allowed_source_prefixes = ["tables", "s3://bucket/documents"]

[storage]
aws_region = "us-east-1"

[session]
"parqdb.execution.query_dop" = "8"
""",
        encoding="utf-8",
    )

    configuration = load_server_config(path)

    assert configuration.root == (tmp_path / "state").resolve()
    assert configuration.allowed_source_prefixes == (
        str((tmp_path / "tables").resolve()),
        "s3://bucket/documents",
    )
    assert dict(configuration.storage_options) == {"aws_region": "us-east-1"}
    assert configuration.session_config() is not None


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("[server]\nunknown = true\n", "unsupported key"),
        (
            '[server]\ncatalog = "sqlite:///tmp/a.sqlite"\n',
            "unsupported key",
        ),
        ("[server]\nport = 0\n", "port must be an integer"),
        ("[storage]\naws_region = 1\n", "must be a string"),
    ],
)
def test_server_config_rejects_ambiguous_or_invalid_values(
    tmp_path: Path,
    document: str,
    message: str,
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_server_config(path)


def test_server_config_allows_a_separate_warehouse(tmp_path: Path) -> None:
    path = tmp_path / "remote.toml"
    path.write_text(
        """
[server]
root = "state"
warehouse = "s3://bucket/parqdb/"
""",
        encoding="utf-8",
    )

    configuration = load_server_config(path)

    assert configuration.root == (tmp_path / "state").resolve()
    assert configuration.warehouse == "s3://bucket/parqdb/"


def test_config_init_writes_template_without_overwriting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "parqdb.toml"

    assert main(["config", "init", "--path", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE
    assert str(path.resolve()) in capsys.readouterr().out

    with pytest.raises(SystemExit, match="2"):
        main(["config", "init", "--path", str(path)])


def test_serve_uses_default_config_only_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    defaults = _configuration_for_serve(None)

    assert defaults.root == (tmp_path / "parqdb").resolve()
    assert "using built-in defaults" in capsys.readouterr().err

    (tmp_path / "parqdb.toml").write_text("[server]\nport = 9000\n", encoding="utf-8")
    configured = _configuration_for_serve(None)

    assert configured.port == 9000


def test_serve_builds_one_worker_from_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "parqdb.toml"
    path.write_text(
        """
[server]
root = "state"
port = 9001
warehouse = "s3://bucket/parqdb/"

[storage]
aws_region = "us-east-1"
""",
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    def create_app(*args: object, **kwargs: object) -> object:
        calls["app"] = (args, kwargs)
        return "app"

    def run(app: object, **kwargs: object) -> None:
        calls["run"] = (app, kwargs)

    monkeypatch.setattr("parqdb.cli.create_app", create_app)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))

    assert main(["serve", "--config", str(path)]) == 0

    assert calls["app"] == (
        ((tmp_path / "state").resolve(),),
        {
            "warehouse": "s3://bucket/parqdb/",
            "storage_options": {"aws_region": "us-east-1"},
            "config": None,
            "allowed_source_prefixes": (),
        },
    )
    assert calls["run"] == (
        "app",
        {"host": "127.0.0.1", "port": 9001, "workers": 1},
    )

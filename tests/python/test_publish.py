from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from parqdb.cli import main
from parqdb.publish import publish

FIXTURE = Path("spec/fixtures/v1/valid/lvq8")


def _inputs(tmp_path: Path, keys: list[int] | None = None) -> tuple[Path, Path]:
    package = tmp_path / "package"
    shutil.copytree(FIXTURE, package)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index"]["source-key-fields"] = [{"name": "chunk_id", "type": "long"}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source = tmp_path / "documents.parquet"
    schema = pa.schema(
        [
            pa.field("chunk_id", pa.int64(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array(keys or [0, 1, 2]), pa.array(["a", "b", "c"])], schema=schema
        ),
        source,
        row_group_size=2,
    )
    return manifest_path, source


def test_publish_local_source_and_index(tmp_path: Path) -> None:
    manifest, source = _inputs(tmp_path)
    destination = tmp_path / "site" / "data"

    result = publish(
        index_manifest=manifest,
        source=source,
        source_key="chunk_id",
        destination=str(destination),
    )

    assert result.destination == str(destination)
    assert result.manifest_url is None
    assert (
        destination / "index" / "manifest.json"
    ).read_bytes() == manifest.read_bytes()
    source_manifest = json.loads(
        (destination / "source-manifest.json").read_text(encoding="utf-8")
    )
    assert source_manifest["key"] == {"name": "chunk_id", "type": "long"}
    assert source_manifest["rows"] == 3
    assert source_manifest["row-group-rows"] == 2
    assert source_manifest["object"]["path"] == "documents.parquet"
    assert (destination / "index" / "ivf_postings" / "manifest.json").is_file()

    with pytest.raises(FileExistsError, match="destination already exists"):
        publish(
            index_manifest=manifest,
            source=source,
            source_key="chunk_id",
            destination=str(destination),
        )


def test_publish_rejects_source_key_that_is_not_dense_and_ordered(
    tmp_path: Path,
) -> None:
    manifest, source = _inputs(tmp_path, [0, 2, 3])

    with pytest.raises(ValueError, match="expected 1, got 2"):
        publish(
            index_manifest=manifest,
            source=source,
            source_key="chunk_id",
            destination=str(tmp_path / "output"),
        )


def test_publish_cli_accepts_an_existing_static_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, source = _inputs(tmp_path)
    destination = tmp_path / "output"

    assert (
        main(
            [
                "publish",
                "--source",
                str(source),
                "--key",
                "chunk_id",
                "--index-manifest",
                str(manifest),
                "--destination",
                str(destination),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["destination"] == str(destination)
    assert output["manifest_url"] is None
    assert output["files"] >= 6

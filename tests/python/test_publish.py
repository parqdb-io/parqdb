from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from parqdb.cli import main
from parqdb.publish import build_index, publish

FIXTURE = Path(__file__).parents[2] / "spec" / "fixtures" / "v1" / "valid" / "lvq8"


def _inputs(tmp_path: Path, keys: list[int] | None = None) -> tuple[Path, Path]:
    artifact = tmp_path / "artifact"
    shutil.copytree(FIXTURE, artifact)
    manifest_path = artifact / "manifest.json"
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
    publication = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert publication["source"]["key"] == {"name": "chunk_id", "type": "long"}
    assert publication["source"]["rows"] == 3
    assert publication["source"]["row-group-rows"] == 2
    assert publication["source"]["files"][0]["path"] == "documents.parquet"
    assert not (destination / "source-manifest.json").exists()
    assert not (destination / "ivf_postings" / "manifest.json").exists()

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


def test_publish_reports_a_missing_source_key_cleanly(tmp_path: Path) -> None:
    manifest, source = _inputs(tmp_path)

    with pytest.raises(ValueError, match="source key column does not exist: missing"):
        publish(
            index_manifest=manifest,
            source=source,
            source_key="missing",
            destination=str(tmp_path / "output"),
        )


def test_build_work_cannot_silently_reuse_a_different_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "vectors.parquet"
    schema = pa.schema(
        [
            pa.field("chunk_id", pa.int64(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), 2), nullable=False),
        ]
    )

    def write(values: list[list[float]]) -> None:
        pq.write_table(
            pa.Table.from_arrays(
                [pa.array([0, 1]), pa.array(values, type=schema.field(1).type)],
                schema=schema,
            ),
            source,
        )

    artifact = tmp_path / "artifact"
    shutil.copytree(FIXTURE, artifact)
    monkeypatch.setattr(
        "parqdb.publish._build_parqdb_index", lambda *args, **kwargs: artifact
    )
    work = tmp_path / "work"
    write([[1.0, 0.0], [0.0, 1.0]])
    build_index(
        source=source,
        source_key="chunk_id",
        work=work,
        nlist=2,
        encoding="lvq8",
        metric="cosine",
        threads=1,
        vector_column="embedding",
    )
    write([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="work directory belongs to a different build"):
        build_index(
            source=source,
            source_key="chunk_id",
            work=work,
            nlist=2,
            encoding="lvq8",
            metric="cosine",
            threads=1,
            vector_column="embedding",
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
    assert output["files"] == 3
    published = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert "source" not in published

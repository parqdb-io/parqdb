from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import parqdb
import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq
import pytest
from _support import WAIT, drop_table_index_entry, register_source, vector_type
from parqdb.publish import build_index, publish
from support.config import S3Config

pytestmark = pytest.mark.requires("s3")


def _s3_filesystem(config: S3Config) -> fs.S3FileSystem:
    endpoint = urlsplit(config.endpoint)
    assert endpoint.hostname is not None
    endpoint_override = endpoint.hostname
    if endpoint.port is not None:
        endpoint_override = f"{endpoint_override}:{endpoint.port}"
    return fs.S3FileSystem(
        access_key=config.access_key,
        secret_key=config.secret_key,
        endpoint_override=endpoint_override,
        scheme=endpoint.scheme,
        region=config.region,
        allow_bucket_creation=True,
    )


def _write_source(
    filesystem: fs.S3FileSystem,
    path: str,
    rows: int,
) -> None:
    vectors = [[float(value), 0.0] for value in range(rows)]
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("payload", pa.string(), nullable=False),
            pa.field("embedding", vector_type(), nullable=False),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array(range(rows), type=pa.int64()),
            pa.array([f"row-{value}" for value in range(rows)]),
            pa.array(vectors, type=vector_type()),
        ],
        schema=schema,
    )
    with filesystem.open_output_stream(path) as sink:
        pq.write_table(table, sink)


def test_s3_build_search_refresh_and_gc(
    tmp_path: Path,
    s3: S3Config,
) -> None:
    filesystem = _s3_filesystem(s3)
    base_uri = f"{s3.uri}/parqdb-{uuid.uuid4().hex}"
    base = urlsplit(base_uri)
    base_path = f"{base.netloc}/{base.path.strip('/')}"
    source_path = f"{base_path}/source/documents.parquet"
    source_uri = f"{base_uri}/source/documents.parquet"
    index_root = f"{base_uri}/warehouse"
    filesystem.create_dir(f"{base_path}/source", recursive=True)
    _write_source(filesystem, source_path, 4)

    session = parqdb.connect(
        tmp_path / "state",
        warehouse=index_root,
        storage_options=s3.storage_options,
    )
    documents = register_source(
        session,
        source_uri,
        "documents",
    )
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=2),
        wait_timeout=WAIT,
    )

    first = session.collect(documents.search([0.0, 0.0]).nprobes(2).limit(4))
    assert first["id"].to_pylist() == [0, 1, 2, 3]
    assert first["_distance"].to_pylist() == [0.0, 1.0, 4.0, 9.0]

    _write_source(filesystem, source_path, 5)
    documents.refresh_index("documents_embedding", wait_timeout=WAIT)
    refreshed = session.collect(documents.search([4.0, 0.0]).nprobes(2).limit(1))
    assert refreshed["id"].to_pylist() == [4]
    assert refreshed["_distance"].to_pylist() == [0.0]

    drop_table_index_entry(session, documents, "documents_embedding")
    cutoff = datetime.now(UTC) - timedelta(days=7)
    assert session.maintenance.remove_orphans(older_than=cutoff) == ()
    with sqlite3.connect(session.root / "catalog.sqlite") as connection:
        connection.execute(
            "UPDATE catalog_tombstones SET unreachable_since_ms = ?",
            (int((datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000),),
        )
    candidates = session.maintenance.remove_orphans(older_than=cutoff)
    assert {candidate.kind for candidate in candidates} == {
        "metadata",
        "index_data",
    }
    removed = session.maintenance.remove_orphans(
        older_than=cutoff,
        dry_run=False,
    )
    assert removed == candidates

    source = filesystem.get_file_info(source_path)
    assert source.type == fs.FileType.File
    warehouse = filesystem.get_file_info(f"{base_path}/warehouse")
    assert warehouse.type == fs.FileType.NotFound
    filesystem.delete_dir(base_path)


def test_publish_static_index_to_s3(
    tmp_path: Path,
    s3: S3Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "documents.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), 2), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(range(4), type=pa.int64()),
                pa.array(
                    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
                    type=schema.field("embedding").type,
                ),
            ],
            schema=schema,
        ),
        source,
    )
    built = build_index(
        source=source,
        source_key="id",
        work=tmp_path / "work",
        nlist=2,
        encoding="lvq8",
        metric="cosine",
        threads=1,
        vector_column="embedding",
    )

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", s3.access_key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", s3.secret_key)
    destination = f"{s3.uri}/publication-{uuid.uuid4().hex}/v1"
    result = publish(
        index_manifest=built.manifest,
        source=source,
        source_key="id",
        destination=destination,
        s3_endpoint=s3.endpoint,
        s3_region=s3.region,
    )

    filesystem = _s3_filesystem(s3)
    parsed = urlsplit(destination)
    prefix = f"{parsed.netloc}/{parsed.path.strip('/')}"
    try:
        assert result.destination == destination
        assert result.files > 1
        assert (
            filesystem.get_file_info(f"{prefix}/manifest.json").type == fs.FileType.File
        )
        assert (
            filesystem.get_file_info(f"{prefix}/documents.parquet").type
            == fs.FileType.File
        )
    finally:
        filesystem.delete_dir(prefix)

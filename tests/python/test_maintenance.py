from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse

import parqdb
import pytest
from _support import (
    WAIT,
    build_index,
    drop_table_index_entry,
    load_table_index,
    register_source,
    register_table_index,
    write_vectors,
)


def _uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def _set_tree_mtime(root: Path, timestamp: float) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        os.utime(path, (timestamp, timestamp), follow_symlinks=False)
    os.utime(root, (timestamp, timestamp), follow_symlinks=False)


def _set_tombstone_time(session: parqdb.Session, timestamp: datetime) -> None:
    with sqlite3.connect(session.root / "catalog.sqlite") as connection:
        connection.execute(
            "UPDATE catalog_tombstones SET unreachable_since_ms = ?",
            (int(timestamp.timestamp() * 1000),),
        )


def test_remove_orphans_preserves_all_reachable_objects(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    old = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    _set_tree_mtime(session.root / "metadata", old)
    _set_tree_mtime(session.root / "indexes", old)

    candidates = session.maintenance.remove_orphans(
        older_than=datetime.now(UTC) - timedelta(days=7),
    )

    assert candidates == ()


def test_refresh_exposes_only_superseded_metadata_as_an_orphan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    before = load_table_index(session, vectors, "vectors_embedding")
    old = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    os.utime(_uri_path(before.metadata_location), (old, old))

    vectors.refresh_index("vectors_embedding", wait_timeout=WAIT)
    cutoff = datetime.now(UTC) - timedelta(days=7)
    assert (
        session.maintenance.remove_orphans(
            older_than=cutoff,
        )
        == ()
    )

    _set_tombstone_time(session, datetime.now(UTC) - timedelta(days=30))
    candidates = session.maintenance.remove_orphans(
        older_than=cutoff,
    )

    assert len(candidates) == 1
    assert candidates[0].kind == "metadata"
    assert candidates[0].reference == before.metadata_location
    assert _uri_path(before.metadata_location).is_file()


def test_remove_orphans_dry_run_and_delete_dropped_index_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    old = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    _set_tree_mtime(session.root / "metadata", old)
    _set_tree_mtime(session.root / "indexes", old)
    drop_table_index_entry(session, vectors, "vectors_embedding")
    cutoff = datetime.now(UTC) - timedelta(days=7)

    assert session.maintenance.remove_orphans(older_than=cutoff) == ()
    _set_tombstone_time(session, datetime.now(UTC) - timedelta(days=30))
    candidates = session.maintenance.remove_orphans(older_than=cutoff)

    assert {candidate.kind for candidate in candidates} == {
        "metadata",
        "index_data",
    }
    assert all(_uri_path(candidate.reference).exists() for candidate in candidates)
    removed = session.maintenance.remove_orphans(
        older_than=cutoff,
        dry_run=False,
    )
    assert removed == candidates
    assert all(not _uri_path(candidate.reference).exists() for candidate in removed)
    assert source.is_file()
    assert (session.root / "catalog.sqlite").is_file()


def test_removed_metadata_cannot_be_registered_from_session_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    entry = load_table_index(session, vectors, "vectors_embedding")
    drop_table_index_entry(session, vectors, "vectors_embedding")
    _set_tombstone_time(session, datetime.now(UTC) - timedelta(days=30))

    removed = session.maintenance.remove_orphans(
        older_than=datetime.now(UTC),
        dry_run=False,
    )

    assert any(candidate.kind == "metadata" for candidate in removed)
    assert not _uri_path(entry.metadata_location).exists()
    with pytest.raises(parqdb.StorageError):
        register_table_index(
            vectors,
            "resurrected",
            entry.metadata_location,
        )
    assert session._indexes.list(namespace=vectors.identifier.index_namespace) == []


def test_lazy_query_is_protected_by_retention_without_a_query_lease(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    context = session.datafusion_context()
    df = context.sql(session.to_sql(vectors.search([0.0, 0.0]).limit(2)))
    df = df.filter("id >= 0").select("id", "_distance").limit(1)
    old = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    _set_tree_mtime(session.root / "metadata", old)
    _set_tree_mtime(session.root / "indexes", old)
    drop_table_index_entry(session, vectors, "vectors_embedding")

    first_removed = session.maintenance.remove_orphans(
        older_than=datetime.now(UTC),
        dry_run=False,
    )

    assert first_removed == ()
    assert not (session.root / ".locks" / "query-leases").exists()
    assert df.collect()[0]["id"].to_pylist() == [0]

    _set_tombstone_time(session, datetime.now(UTC) - timedelta(days=30))
    second_removed = session.maintenance.remove_orphans(
        older_than=datetime.now(UTC),
        dry_run=False,
    )
    assert any(candidate.kind == "index_data" for candidate in second_removed)


def test_temp_view_remains_readable_during_retention(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    context = session.datafusion_context()
    context.register_view(
        "hits",
        context.sql(session.to_sql(vectors.search([0.0, 0.0]).limit(1).select(["id"]))),
    )
    old = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    _set_tree_mtime(session.root / "metadata", old)
    _set_tree_mtime(session.root / "indexes", old)
    drop_table_index_entry(session, vectors, "vectors_embedding")

    first_removed = session.maintenance.remove_orphans(
        older_than=datetime.now(UTC),
        dry_run=False,
    )

    assert first_removed == ()
    assert context.sql("SELECT id FROM hits").to_pydict() == {"id": [0]}

    context.deregister_table("hits")
    _set_tombstone_time(session, datetime.now(UTC) - timedelta(days=30))
    second_removed = session.maintenance.remove_orphans(
        older_than=datetime.now(UTC),
        dry_run=False,
    )
    assert any(candidate.kind == "index_data" for candidate in second_removed)


def test_remove_orphans_ignores_recent_and_unknown_objects(tmp_path: Path) -> None:
    session = parqdb.connect(tmp_path / "parqdb-data")
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    drop_table_index_entry(session, vectors, "vectors_embedding")
    metadata_unknown = (
        session.root
        / "metadata"
        / "2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1"
        / "do-not-touch.txt"
    )
    metadata_unknown.parent.mkdir(parents=True, exist_ok=True)
    metadata_unknown.write_text("caller data")
    index_unknown = (
        session.root / "indexes" / "2f1c7f5e3c434a448f2acf560c4db8d1" / "not-a-snapshot"
    )
    index_unknown.mkdir(parents=True)
    old = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    _set_tree_mtime(metadata_unknown.parent, old)
    _set_tree_mtime(index_unknown, old)

    candidates = session.maintenance.remove_orphans(
        older_than=datetime.now(UTC) - timedelta(days=1),
        dry_run=False,
    )

    assert candidates == ()
    assert metadata_unknown.is_file()
    assert index_unknown.is_dir()


def test_remove_orphans_enforces_minimum_retention(
    tmp_path: Path,
) -> None:
    session = parqdb.connect(tmp_path / "parqdb-data")
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    old = (datetime.now(UTC) - timedelta(days=30)).timestamp()
    _set_tree_mtime(session.root / "metadata", old)
    _set_tree_mtime(session.root / "indexes", old)
    drop_table_index_entry(session, vectors, "vectors_embedding")
    cutoff = datetime.now(UTC)

    assert session.maintenance.remove_orphans(older_than=cutoff) == ()
    _set_tombstone_time(session, datetime.now(UTC) - timedelta(days=30))
    assert {
        candidate.kind
        for candidate in session.maintenance.remove_orphans(
            older_than=cutoff,
        )
    } == {"metadata", "index_data"}


@pytest.mark.parametrize("value", [None, "yesterday", 1])
def test_remove_orphans_requires_a_datetime(
    tmp_path: Path,
    value: object,
) -> None:
    session = parqdb.connect(tmp_path / "parqdb-data")

    with pytest.raises(TypeError, match="must be a datetime"):
        session.maintenance.remove_orphans(older_than=cast(Any, value))


def test_remove_orphans_validates_datetime_and_dry_run(tmp_path: Path) -> None:
    session = parqdb.connect(tmp_path / "parqdb-data")

    with pytest.raises(ValueError, match="timezone-aware"):
        session.maintenance.remove_orphans(older_than=datetime.now())
    with pytest.raises(ValueError, match="must not be in the future"):
        session.maintenance.remove_orphans(
            older_than=datetime.now(UTC) + timedelta(days=1),
        )
    with pytest.raises(TypeError, match="dry_run must be a boolean"):
        session.maintenance.remove_orphans(
            older_than=datetime.now(UTC),
            dry_run=cast(Any, 1),
        )

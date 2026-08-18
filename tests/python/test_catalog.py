from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import parqdb
import pytest
from _support import (
    drop_table_index_entry,
    load_table_index,
    register_source,
    register_table_index,
    relation_files,
    write_vectors,
)
from parqdb.runtime.catalog import open_index_catalog


def test_catalog_lifecycle_survives_session_reopen(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )
    session = parqdb.connect(tmp_path / "parqdb-data")
    assert session._indexes is not None
    vectors = register_source(session, source)
    vectors.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=2),
    )

    assert vectors.index_status("vectors_embedding").state in {
        "pending",
        "building",
        "ready",
    }
    vectors.wait_for_index(
        "vectors_embedding",
        timeout=timedelta(seconds=30),
    )
    assert vectors.index_status("vectors_embedding").state == "ready"
    namespace = vectors.identifier.index_namespace
    assert session._indexes.list(namespace=namespace) == ["vectors_embedding"]

    entry = load_table_index(session, vectors, "vectors_embedding")
    assert entry.identifier == "vectors_embedding"
    assert isinstance(entry.metadata["snapshots"], tuple)
    assert set(entry.metadata["snapshots"][0]["index-relations"]) == {
        "ivf_centroids",
        "ivf_postings",
    }
    source_reference = {"profile": "parquet", "uri": source.as_uri()}
    assert session._indexes.list_for(source_reference, namespace=namespace) == [
        parqdb.IndexInfo(
            name="vectors_embedding",
            column="embedding",
            family="ivf",
            metric="l2_squared",
            parameters=entry.metadata["snapshots"][0]["parameters"],
            current_snapshot_id=entry.metadata["current-snapshot-id"],
        )
    ]
    assert session._indexes.select(source_reference, namespace=namespace) == entry
    with pytest.raises(TypeError):
        entry.metadata["current-snapshot-id"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        entry.metadata["snapshots"][0]["metric"] = "other"

    reopened = parqdb.connect(tmp_path / "parqdb-data")
    reopened_vectors = reopened.table("vectors")
    assert isinstance(reopened_vectors, parqdb.SourceTable)
    assert reopened._indexes.list(namespace=namespace) == ["vectors_embedding"]
    assert load_table_index(reopened, reopened_vectors, "vectors_embedding") == entry


def test_internal_index_catalog_can_be_opened_without_an_execution_session(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    root = tmp_path / "state"
    catalog_path = root / "catalog.sqlite"
    warehouse = (tmp_path / "warehouse").as_uri()
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(
        root,
        warehouse=warehouse,
    )
    vectors = register_source(session, source)
    vectors.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=1),
        wait_timeout=timedelta(seconds=30),
    )
    expected = load_table_index(session, vectors, "vectors_embedding")
    namespace = vectors.identifier.index_namespace

    catalog = open_index_catalog(
        f"sqlite://{catalog_path}",
        warehouse=warehouse,
    )

    assert catalog.load("vectors_embedding", namespace=namespace) == expected
    source_reference = {"profile": "parquet", "uri": source.as_uri()}
    assert (
        catalog.select(source_reference, namespace=namespace).identifier
        == "vectors_embedding"
    )


def test_catalog_load_reports_a_missing_index(tmp_path: Path) -> None:
    session = parqdb.connect(tmp_path / "parqdb-data")

    with pytest.raises(parqdb.IndexNotFoundError):
        session._indexes.load("missing")


def test_catalog_drop_and_register_recover_an_existing_index(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    vectors.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=1),
        wait_timeout=timedelta(seconds=30),
    )
    entry = load_table_index(session, vectors, "vectors_embedding")
    index_files = [
        path
        for reference in entry.metadata["snapshots"][0]["index-relations"].values()
        for path in relation_files(reference, session.warehouse)
    ]

    drop_table_index_entry(session, vectors, "vectors_embedding")

    assert session._indexes.list(namespace=vectors.identifier.index_namespace) == []
    assert all(path.exists() for path in index_files)
    with pytest.raises(parqdb.IndexNotFoundError):
        vectors.index_status("vectors_embedding")
    with pytest.raises(parqdb.IndexNotFoundError):
        load_table_index(session, vectors, "vectors_embedding")

    register_table_index(
        vectors,
        "vectors_embedding",
        entry.metadata_location,
    )

    assert load_table_index(session, vectors, "vectors_embedding") == entry
    recovered = vectors.index_status("vectors_embedding")
    assert recovered.state == "ready"
    assert recovered.current_snapshot_id == entry.metadata["current-snapshot-id"]
    with pytest.raises(parqdb.AlreadyExistsError):
        register_table_index(
            vectors,
            "vectors_embedding",
            entry.metadata_location,
        )


def test_catalogs_share_indexes_through_one_warehouse(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    warehouse = (tmp_path / "warehouse").as_uri()
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])

    publisher = parqdb.connect(tmp_path / "publisher", warehouse=warehouse)
    published_vectors = register_source(publisher, source)
    published_vectors.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=1),
        wait_timeout=timedelta(seconds=30),
    )
    published = load_table_index(
        publisher,
        published_vectors,
        "vectors_embedding",
    )

    consumer = parqdb.connect(tmp_path / "consumer", warehouse=warehouse)
    consumed_vectors = register_source(consumer, source)
    register_table_index(
        consumed_vectors,
        "vectors_embedding",
        published.metadata_location,
    )

    consumed = load_table_index(consumer, consumed_vectors, "vectors_embedding")
    assert consumed == published
    assert consumer.to_arrow(consumed_vectors.search([0.0, 0.0]).limit(1))[
        "id"
    ].to_pylist() == [0]
    assert (tmp_path / "publisher" / "catalog.sqlite").is_file()
    assert (tmp_path / "consumer" / "catalog.sqlite").is_file()


def test_index_status_and_wait_are_scoped_to_the_source(tmp_path: Path) -> None:
    first_source = tmp_path / "first.parquet"
    second_source = tmp_path / "second.parquet"
    write_vectors(first_source, [0], [[0.0, 0.0]])
    write_vectors(second_source, [0], [[1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    first = register_source(session, first_source, "first")
    second = register_source(session, second_source, "second")
    first.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=1),
        wait_timeout=timedelta(seconds=30),
    )

    assert first.index_status("vectors_embedding").state == "ready"
    with pytest.raises(parqdb.IndexNotFoundError):
        second.index_status("vectors_embedding")
    with pytest.raises(parqdb.IndexNotFoundError):
        second.wait_for_index(
            "vectors_embedding",
            timeout=timedelta(seconds=1),
        )

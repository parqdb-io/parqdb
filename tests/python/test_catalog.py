from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import relify
from _support import (
    drop_table_index_entry,
    load_table_index,
    register_source,
    register_table_index,
    relation_files,
    write_vectors,
)


def test_catalog_lifecycle_survives_session_reopen(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )
    session = relify.connect(tmp_path / "relify-data")
    assert isinstance(session.indexes, relify.IndexCatalog)
    vectors = register_source(session, source)
    vectors.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=relify.IVF(nlist=2),
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
    assert session.indexes.list(namespace=namespace) == ["vectors_embedding"]

    entry = load_table_index(session, vectors, "vectors_embedding")
    assert entry.identifier == "vectors_embedding"
    assert isinstance(entry.metadata["snapshots"], tuple)
    assert set(entry.metadata["snapshots"][0]["index-relations"]) == {
        "ivf_centroids",
        "ivf_postings",
    }
    source_reference = entry.metadata["snapshots"][0]["source"]
    assert session.indexes.list_for(source_reference, namespace=namespace) == [
        relify.IndexInfo(
            name="vectors_embedding",
            column="embedding",
            family="ivf",
            metric="l2_squared",
            parameters=entry.metadata["snapshots"][0]["parameters"],
            current_snapshot_id=entry.metadata["current-snapshot-id"],
        )
    ]
    assert session.indexes.select(source_reference, namespace=namespace) == entry
    with pytest.raises(TypeError):
        entry.metadata["current-snapshot-id"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        entry.metadata["snapshots"][0]["metric"] = "other"

    reopened = relify.connect(tmp_path / "relify-data")
    reopened_vectors = reopened.table("vectors")
    assert isinstance(reopened_vectors, relify.SourceTable)
    assert reopened.indexes.list(namespace=namespace) == ["vectors_embedding"]
    assert load_table_index(reopened, reopened_vectors, "vectors_embedding") == entry


def test_public_index_catalog_can_be_opened_without_an_execution_session(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    catalog_path = tmp_path / "catalog.sqlite"
    metadata_root = (tmp_path / "metadata").as_uri()
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(
        catalog=f"sqlite://{catalog_path}",
        index_root=metadata_root,
    )
    vectors = register_source(session, source)
    vectors.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=relify.IVF(nlist=1),
        wait_timeout=timedelta(seconds=30),
    )
    expected = load_table_index(session, vectors, "vectors_embedding")
    namespace = vectors.identifier.index_namespace

    catalog = relify.open_index_catalog(
        f"sqlite://{catalog_path}",
        metadata_root=metadata_root,
    )

    assert catalog.load("vectors_embedding", namespace=namespace) == expected
    source_reference = expected.metadata["snapshots"][0]["source"]
    assert (
        catalog.select(source_reference, namespace=namespace).identifier
        == "vectors_embedding"
    )


def test_catalog_load_reports_a_missing_index(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")

    with pytest.raises(relify.IndexNotFoundError):
        session.indexes.load("missing")


def test_catalog_drop_and_register_recover_an_existing_index(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    vectors.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=relify.IVF(nlist=1),
        wait_timeout=timedelta(seconds=30),
    )
    entry = load_table_index(session, vectors, "vectors_embedding")
    index_files = [
        path
        for reference in entry.metadata["snapshots"][0]["index-relations"].values()
        for path in relation_files(reference)
    ]

    drop_table_index_entry(session, vectors, "vectors_embedding")

    assert session.indexes.list(namespace=vectors.identifier.index_namespace) == []
    assert all(path.exists() for path in index_files)
    with pytest.raises(relify.IndexNotFoundError):
        vectors.index_status("vectors_embedding")
    with pytest.raises(relify.IndexNotFoundError):
        load_table_index(session, vectors, "vectors_embedding")

    register_table_index(
        session,
        vectors,
        "vectors_embedding",
        entry.metadata_location,
    )

    assert load_table_index(session, vectors, "vectors_embedding") == entry
    recovered = vectors.index_status("vectors_embedding")
    assert recovered.state == "ready"
    assert recovered.current_snapshot_id == entry.metadata["current-snapshot-id"]
    with pytest.raises(relify.AlreadyExistsError):
        register_table_index(
            session,
            vectors,
            "vectors_embedding",
            entry.metadata_location,
        )


def test_index_status_and_wait_are_scoped_to_the_source(tmp_path: Path) -> None:
    first_source = tmp_path / "first.parquet"
    second_source = tmp_path / "second.parquet"
    write_vectors(first_source, [0], [[0.0, 0.0]])
    write_vectors(second_source, [0], [[1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    first = register_source(session, first_source, "first")
    second = register_source(session, second_source, "second")
    first.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=relify.IVF(nlist=1),
        wait_timeout=timedelta(seconds=30),
    )

    assert first.index_status("vectors_embedding").state == "ready"
    with pytest.raises(relify.IndexNotFoundError):
        second.index_status("vectors_embedding")
    with pytest.raises(relify.IndexNotFoundError):
        second.wait_for_index(
            "vectors_embedding",
            timeout=timedelta(seconds=1),
        )

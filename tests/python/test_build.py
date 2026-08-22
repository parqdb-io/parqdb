from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import parqdb
import pytest
from _support import (
    WAIT,
    artifact_manifest_location,
    build_index,
    drop_table_index_entry,
    load_table_index,
    register_source,
    register_table_index,
    write_vectors,
)


def test_published_index_cannot_be_created_again(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors)

    with pytest.raises(parqdb.AlreadyExistsError, match="index already exists"):
        build_index(vectors)


def test_wait_validation_and_missing_index_errors(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)

    with pytest.raises(ValueError, match="wait_timeout must be positive"):
        vectors.create_index(
            "vectors_embedding",
            column="embedding",
            key=["id"],
            config=parqdb.IVF(nlist=1),
            wait_timeout=timedelta(0),
        )
    assert vectors.list_indexes() == []

    with pytest.raises(ValueError, match="timeout must be positive"):
        vectors.wait_for_index("missing", timeout=timedelta(0))
    with pytest.raises(parqdb.IndexNotFoundError, match="index not found"):
        vectors.wait_for_index("missing", timeout=timedelta(seconds=1))
    with pytest.raises(parqdb.IndexNotFoundError, match="index not found"):
        vectors.index_status("missing")


@pytest.mark.parametrize("name", ["", "1index", "has.dot", "has-hyphen"])
def test_invalid_index_names_fail_as_invalid_arguments(
    tmp_path: Path,
    name: str,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)

    with pytest.raises(parqdb.InvalidArgumentError, match="index name"):
        build_index(vectors, name)

    with pytest.raises(parqdb.InvalidArgumentError, match="index name"):
        vectors.index_status(name)
    assert vectors.list_indexes() == []


@pytest.mark.parametrize(
    ("column", "key", "message"),
    [
        ("", ["id"], "vector column"),
        ("embedding", [], "key must contain"),
        ("embedding", [""], "key must contain"),
        ("embedding", ["id", "id"], "key must contain"),
    ],
)
def test_invalid_build_requests_fail_before_training(
    tmp_path: Path,
    column: str,
    key: list[str],
    message: str,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)

    with pytest.raises(parqdb.InvalidArgumentError, match=message):
        vectors.create_index(
            "vectors_embedding",
            column=column,
            key=key,
            config=parqdb.IVF(nlist=1),
            wait_timeout=WAIT,
        )

    with pytest.raises(parqdb.IndexNotFoundError):
        vectors.index_status("vectors_embedding")
    assert vectors.list_indexes() == []


def test_create_index_rejects_unsupported_python_objects(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)

    with pytest.raises(TypeError, match=r"parqdb\.IVF"):
        vectors.create_index(
            "vectors_embedding",
            column="embedding",
            key=["id"],
            config=object(),  # type: ignore[arg-type]
        )
    assert vectors.list_indexes() == []


def test_failed_build_can_be_retried_under_the_same_name(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)

    with pytest.raises(parqdb.InvalidSchemaError, match="key column not found"):
        build_index(vectors, key=["missing"])
    assert vectors.index_status("vectors_embedding").state == "failed"

    build_index(vectors)
    assert vectors.index_status("vectors_embedding").state == "ready"
    assert [index.name for index in vectors.list_indexes()] == ["vectors_embedding"]


def test_catalog_publication_supersedes_an_unpublished_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)

    with pytest.raises(parqdb.InvalidSchemaError):
        build_index(vectors, "failed_embedding", key=["missing"])
    assert vectors.index_status("failed_embedding").state == "failed"

    build_index(vectors, "published_embedding", encoding="lvq8")
    published = load_table_index(session, vectors, "published_embedding")
    manifest_location = artifact_manifest_location(published, session.warehouse)
    drop_table_index_entry(session, vectors, "published_embedding")
    register_table_index(
        vectors,
        "failed_embedding",
        manifest_location,
    )

    recovered = vectors.index_status("failed_embedding")
    assert recovered.state == "ready"
    assert recovered.current_snapshot_id is not None
    assert recovered.error is None

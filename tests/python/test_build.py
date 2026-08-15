from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from threading import Event

import pyarrow as pa
import pytest
import relify
from _support import WAIT, build_index, register_source, write_vectors
from relify._service import TableDescriptor


def test_published_index_cannot_be_created_again(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors)

    with pytest.raises(relify.AlreadyExistsError, match="index already exists"):
        build_index(vectors)


def test_wait_validation_and_missing_index_errors(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(ValueError, match="wait_timeout must be positive"):
        vectors.create_index(
            "vectors_embedding",
            column="embedding",
            key=["id"],
            config=relify.IVF(nlist=1),
            wait_timeout=timedelta(0),
        )
    assert session.indexes.list() == []

    with pytest.raises(ValueError, match="timeout must be positive"):
        vectors.wait_for_index("missing", timeout=timedelta(0))
    with pytest.raises(relify.IndexNotFoundError, match="index not found"):
        vectors.wait_for_index("missing", timeout=timedelta(seconds=1))
    with pytest.raises(relify.IndexNotFoundError, match="index not found"):
        vectors.index_status("missing")


@pytest.mark.parametrize("name", ["", "1index", "has.dot", "has-hyphen"])
def test_invalid_index_names_fail_as_invalid_arguments(
    tmp_path: Path,
    name: str,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(relify.InvalidArgumentError, match="index name"):
        build_index(vectors, name)

    with pytest.raises(relify.InvalidArgumentError, match="index name"):
        vectors.index_status(name)
    assert session.indexes.list() == []


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
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(relify.InvalidArgumentError, match=message):
        vectors.create_index(
            "vectors_embedding",
            column=column,
            key=key,
            config=relify.IVF(nlist=1),
            wait_timeout=WAIT,
        )

    with pytest.raises(relify.IndexNotFoundError):
        vectors.index_status("vectors_embedding")
    assert session.indexes.list() == []


def test_create_index_rejects_unsupported_python_objects(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(TypeError, match=r"relify\.IVF"):
        vectors.create_index(
            "vectors_embedding",
            column="embedding",
            key=["id"],
            config=object(),  # type: ignore[arg-type]
        )
    assert session.indexes.list() == []


def test_async_coordinator_rejects_duplicate_build_and_times_out(
    tmp_path: Path,
) -> None:
    class BlockingNative:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()
            self.published: tuple[str, str] | None = None

        def index_exists(self, name: str) -> bool:
            return self.published is not None and self.published[1] == name

        def persistent_table_source_by_identifier(
            self,
            _catalog: str,
            _namespace: list[str],
            _name: str,
        ) -> str:
            return "file:///source.parquet"

        def list_source_indexes(
            self,
            source: str,
        ) -> list[tuple[str, str, str, str, dict[str, str], int]]:
            source_uri = json.loads(source)["uri"]
            if self.published is None or self.published[0] != source_uri:
                return []
            return [
                (
                    self.published[1],
                    "embedding",
                    "ivf",
                    "squared_l2",
                    {},
                    1,
                )
            ]

        def load_index_entry(self, name: str) -> tuple[str, str]:
            assert self.published is not None and self.published[1] == name
            return "file:///metadata.json", "{}"

        def create_index(
            self,
            *,
            source: str,
            index_name: str,
            vector_field: str,
            source_key_fields: list[str],
            nlist: int,
            posting_encoding: str,
            metric: str,
            writer_options: object,
            partitions: int | None,
            threads: int | None,
            progress: object | None,
        ) -> str:
            del (
                vector_field,
                source_key_fields,
                nlist,
                posting_encoding,
                metric,
                writer_options,
                partitions,
                threads,
                progress,
            )
            self.started.set()
            assert self.release.wait(timeout=5)
            self.published = (source, index_name)
            return "file:///metadata.json"

    session = relify.connect(tmp_path / "relify-data")
    native = BlockingNative()
    session._native = native  # type: ignore[assignment]
    session._repository = native  # type: ignore[assignment]
    session._indexes = relify.IndexCatalog(native)  # type: ignore[arg-type]
    session._resolve_build_relation = lambda _identifier: {  # type: ignore[method-assign]
        "profile": "parquet",
        "uri": "file:///source.parquet",
    }
    vectors = relify.SourceTable(
        session,
        TableDescriptor(
            relify.TableIdentifier("datafusion", ("public",), "vectors"),
            pa.schema([]),
        ),
    )
    vectors.create_index(
        "vectors_embedding",
        column="embedding",
        key=["id"],
        config=relify.IVF(nlist=1),
    )
    assert native.started.wait(timeout=5)
    status = vectors.index_status("vectors_embedding")
    assert status.phase == "pending"
    assert status.progress == 0.0
    assert status.completed == 0
    assert status.total == 0
    try:
        with pytest.raises(relify.BuildAlreadyRunningError):
            vectors.create_index(
                "vectors_embedding",
                column="embedding",
                key=["id"],
                config=relify.IVF(nlist=1),
            )
        with pytest.raises(TimeoutError, match="timed out waiting"):
            vectors.wait_for_index(
                "vectors_embedding",
                timeout=timedelta(microseconds=1),
            )
    finally:
        native.release.set()
    vectors.wait_for_index("vectors_embedding", timeout=WAIT)


def test_independent_sessions_do_not_share_a_build_queue(tmp_path: Path) -> None:
    class BlockingNative:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()
            self.published: tuple[str, str] | None = None

        def index_exists(self, name: str) -> bool:
            return self.published is not None and self.published[1] == name

        def persistent_table_source_by_identifier(
            self,
            _catalog: str,
            _namespace: list[str],
            name: str,
        ) -> str:
            return f"file:///{name}.parquet"

        def list_source_indexes(
            self,
            source: str,
        ) -> list[tuple[str, str, str, str, dict[str, str], int]]:
            source_uri = json.loads(source)["uri"]
            if self.published is None or self.published[0] != source_uri:
                return []
            return [
                (
                    self.published[1],
                    "embedding",
                    "ivf",
                    "squared_l2",
                    {},
                    1,
                )
            ]

        def load_index_entry(self, name: str) -> tuple[str, str]:
            assert self.published is not None and self.published[1] == name
            return "file:///metadata.json", "{}"

        def create_index(
            self,
            *,
            source: str,
            index_name: str,
            vector_field: str,
            source_key_fields: list[str],
            nlist: int,
            posting_encoding: str,
            metric: str,
            writer_options: object,
            partitions: int | None,
            threads: int | None,
            progress: object | None,
        ) -> str:
            del (
                vector_field,
                source_key_fields,
                nlist,
                posting_encoding,
                metric,
                writer_options,
                partitions,
                threads,
                progress,
            )
            self.started.set()
            assert self.release.wait(timeout=5)
            self.published = (source, index_name)
            return "file:///metadata.json"

    first_session = relify.connect(tmp_path / "first")
    second_session = relify.connect(tmp_path / "second")
    first_native = BlockingNative()
    second_native = BlockingNative()
    first_session._native = first_native  # type: ignore[assignment]
    second_session._native = second_native  # type: ignore[assignment]
    first_session._repository = first_native  # type: ignore[assignment]
    second_session._repository = second_native  # type: ignore[assignment]
    first_session._indexes = relify.IndexCatalog(first_native)  # type: ignore[arg-type]
    second_session._indexes = relify.IndexCatalog(second_native)  # type: ignore[arg-type]
    first_session._resolve_build_relation = lambda _identifier: {  # type: ignore[method-assign]
        "profile": "parquet",
        "uri": "file:///first.parquet",
    }
    second_session._resolve_build_relation = lambda _identifier: {  # type: ignore[method-assign]
        "profile": "parquet",
        "uri": "file:///second.parquet",
    }
    first = relify.SourceTable(
        first_session,
        TableDescriptor(
            relify.TableIdentifier("datafusion", ("public",), "first"),
            pa.schema([]),
        ),
    )
    second = relify.SourceTable(
        second_session,
        TableDescriptor(
            relify.TableIdentifier("datafusion", ("public",), "second"),
            pa.schema([]),
        ),
    )

    try:
        first.create_index(
            "first_embedding",
            column="embedding",
            key=["id"],
            config=relify.IVF(nlist=1),
        )
        second.create_index(
            "second_embedding",
            column="embedding",
            key=["id"],
            config=relify.IVF(nlist=1),
        )
        assert first_native.started.wait(timeout=5)
        assert second_native.started.wait(timeout=5)
    finally:
        first_native.release.set()
        second_native.release.set()
    first.wait_for_index("first_embedding", timeout=WAIT)
    second.wait_for_index("second_embedding", timeout=WAIT)


def test_failed_build_can_be_retried_under_the_same_name(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(relify.InvalidSchemaError, match="key column not found"):
        build_index(vectors, key=["missing"])
    assert vectors.index_status("vectors_embedding").state == "failed"

    build_index(vectors)
    assert vectors.index_status("vectors_embedding").state == "ready"
    assert session.indexes.list() == ["vectors_embedding"]


def test_catalog_publication_supersedes_an_unpublished_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(relify.InvalidSchemaError):
        build_index(vectors, "failed_embedding", key=["missing"])
    assert vectors.index_status("failed_embedding").state == "failed"

    build_index(vectors, "published_embedding")
    published = session.indexes.load("published_embedding")
    session.indexes.drop("published_embedding")
    session.indexes.register("failed_embedding", published.metadata_location)

    recovered = vectors.index_status("failed_embedding")
    assert recovered.state == "ready"
    assert recovered.current_snapshot_id == published.metadata["current-snapshot-id"]
    assert recovered.error is None

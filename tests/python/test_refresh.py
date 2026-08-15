from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest
import relify
from _support import WAIT, build_index, register_source, write_vectors


def test_refresh_reuses_ivf_centroids_for_the_same_immutable_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1, 2], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=2)
    before = session.indexes.load("vectors_embedding")

    vectors.refresh_index("vectors_embedding", wait_timeout=WAIT)

    after = session.indexes.load("vectors_embedding")
    snapshots = after.metadata["snapshots"]
    assert after.metadata_location != before.metadata_location
    assert after.metadata["index-uuid"] == before.metadata["index-uuid"]
    assert after.metadata["location"] == before.metadata["location"]
    assert len(snapshots) == 2
    assert snapshots[0] == before.metadata["snapshots"][0]
    assert (
        after.metadata["current-snapshot-id"] != before.metadata["current-snapshot-id"]
    )
    assert snapshots[1]["sequence-number"] == 2
    assert snapshots[1]["summary"]["operation"] == "refresh"
    assert snapshots[1]["parameters"]["nlist"] == "2"
    assert snapshots[1]["parameters"]["ntotal"] == "3"
    for field in (
        "ivf_centroids_fingerprint",
        "ivf_centroids_uuid",
        "ivf_centroids_metadata_location",
    ):
        assert snapshots[1]["parameters"][field] == snapshots[0]["parameters"][field]
    assert (
        snapshots[1]["index-relations"]["ivf_centroids"]
        == snapshots[0]["index-relations"]["ivf_centroids"]
    )
    assert (
        snapshots[1]["index-relations"]["ivf_postings"]
        != snapshots[0]["index-relations"]["ivf_postings"]
    )
    assert session.to_arrow(vectors.search([2.0, 0.0]).limit(1))["id"].to_pylist() == [
        2
    ]
    assert (
        vectors.index_status("vectors_embedding").current_snapshot_id
        == (after.metadata["current-snapshot-id"])
    )


def test_refresh_can_change_ivf_configuration(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=2)

    vectors.refresh_index(
        "vectors_embedding",
        config=relify.IVF(nlist=1, encoding="source"),
    )
    vectors.wait_for_index("vectors_embedding", timeout=WAIT)

    metadata = session.indexes.load("vectors_embedding").metadata
    assert metadata["snapshots"][0]["parameters"]["nlist"] == "2"
    assert metadata["snapshots"][1]["parameters"]["nlist"] == "1"
    assert metadata["snapshots"][0]["parameters"]["posting_encoding"] == "source"
    assert metadata["snapshots"][1]["parameters"]["posting_encoding"] == "source"
    assert (
        metadata["snapshots"][0]["parameters"]["ivf_centroids_fingerprint"]
        != metadata["snapshots"][1]["parameters"]["ivf_centroids_fingerprint"]
    )


def test_refresh_rejects_changing_the_distance_metric(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[1.0, 0.0], [0.0, 1.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    before = session.indexes.load("vectors_embedding")

    with pytest.raises(
        relify.InvalidArgumentError,
        match="refresh cannot change the distance metric",
    ):
        vectors.refresh_index(
            "vectors_embedding",
            config=relify.IVF(nlist=1, metric="cosine"),
            wait_timeout=WAIT,
        )

    after = session.indexes.load("vectors_embedding")
    assert after.metadata_location == before.metadata_location
    assert after.metadata == before.metadata


def test_failed_refresh_preserves_the_published_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    before = session.indexes.load("vectors_embedding")

    with pytest.raises(relify.RelifyError):
        vectors.refresh_index(
            "vectors_embedding",
            config=relify.IVF(nlist=3),
            wait_timeout=WAIT,
        )

    after = session.indexes.load("vectors_embedding")
    status = vectors.index_status("vectors_embedding")
    assert after.metadata_location == before.metadata_location
    assert after.metadata == before.metadata
    assert status.state == "ready"
    assert status.current_snapshot_id == before.metadata["current-snapshot-id"]
    assert status.error is not None
    assert session.to_arrow(vectors.search([0.0, 0.0]).limit(1))["id"].to_pylist() == [
        0
    ]

    reopened = relify.connect(tmp_path / "relify-data")
    reopened_vectors = reopened.table("vectors")
    assert isinstance(reopened_vectors, relify.SourceTable)
    reopened_vectors.refresh_index("vectors_embedding", wait_timeout=WAIT)

    refreshed = vectors.index_status("vectors_embedding")
    assert refreshed.state == "ready"
    assert refreshed.current_snapshot_id != before.metadata["current-snapshot-id"]
    assert refreshed.error is None


def test_refresh_keeps_the_previous_snapshot_visible_while_building(
    tmp_path: Path,
) -> None:
    class BlockingRefresh:
        def __init__(self, delegate: Any) -> None:
            self._delegate = delegate
            self.started = Event()
            self.release = Event()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._delegate, name)

        def refresh_index(
            self,
            *,
            source: str,
            index_name: str,
            nlist: int | None,
            posting_encoding: str | None,
            metric: str | None,
            writer_options: object,
            partitions: int | None,
            threads: int | None,
            progress: object | None,
        ) -> str:
            self.started.set()
            assert self.release.wait(timeout=5)
            return self._delegate.refresh_index(
                source=source,
                index_name=index_name,
                nlist=nlist,
                posting_encoding=posting_encoding,
                metric=metric,
                writer_options=writer_options,
                partitions=partitions,
                threads=threads,
                progress=progress,
            )

    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    previous_snapshot_id = vectors.index_status("vectors_embedding").current_snapshot_id
    blocking = BlockingRefresh(session._native)
    session._native = blocking  # type: ignore[assignment]

    vectors.refresh_index("vectors_embedding")
    assert blocking.started.wait(timeout=5)
    try:
        status = vectors.index_status("vectors_embedding")
        assert status.state == "building"
        assert status.current_snapshot_id == previous_snapshot_id
        assert session.to_arrow(vectors.search([0.0, 0.0]).limit(1))[
            "id"
        ].to_pylist() == [0]
    finally:
        blocking.release.set()
    vectors.wait_for_index("vectors_embedding", timeout=WAIT)
    assert (
        vectors.index_status("vectors_embedding").current_snapshot_id
        != previous_snapshot_id
    )


def test_refresh_requires_an_index_for_the_bound_source(tmp_path: Path) -> None:
    first_source = tmp_path / "first.parquet"
    second_source = tmp_path / "second.parquet"
    write_vectors(first_source, [0], [[0.0, 0.0]])
    write_vectors(second_source, [0], [[0.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    first = register_source(session, first_source, "first")
    second = register_source(session, second_source, "second")
    build_index(first, nlist=1)

    with pytest.raises(relify.IndexNotFoundError):
        second.refresh_index("vectors_embedding")
    with pytest.raises(relify.IndexNotFoundError):
        first.refresh_index("missing")


def test_refresh_validates_config_and_timeout(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(TypeError, match=r"relify\.IVF"):
        vectors.refresh_index("missing", config=cast(Any, object()))
    with pytest.raises(ValueError, match="wait_timeout must be positive"):
        vectors.refresh_index("missing", wait_timeout=timedelta(0))

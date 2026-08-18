from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import parqdb
import pytest
from _support import WAIT, build_index, load_table_index, register_source, write_vectors


def test_refresh_reuses_ivf_centroids_for_the_same_immutable_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1, 2], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=2)
    before = load_table_index(session, vectors, "vectors_embedding")

    vectors.refresh_index("vectors_embedding", wait_timeout=WAIT)

    after = load_table_index(session, vectors, "vectors_embedding")
    snapshots = after.metadata["snapshots"]
    assert after.metadata_location != before.metadata_location
    assert after.metadata["index-uuid"] == before.metadata["index-uuid"]
    assert "location" not in after.metadata
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
    assert session.collect(vectors.search([2.0, 0.0]).limit(1))["id"].to_pylist() == [2]
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
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=2)

    vectors.refresh_index(
        "vectors_embedding",
        config=parqdb.IVF(nlist=1, encoding="source"),
    )
    vectors.wait_for_index("vectors_embedding", timeout=WAIT)

    metadata = load_table_index(session, vectors, "vectors_embedding").metadata
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
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    before = load_table_index(session, vectors, "vectors_embedding")

    with pytest.raises(
        parqdb.InvalidArgumentError,
        match="refresh cannot change the distance metric",
    ):
        vectors.refresh_index(
            "vectors_embedding",
            config=parqdb.IVF(nlist=1, metric="cosine"),
            wait_timeout=WAIT,
        )

    after = load_table_index(session, vectors, "vectors_embedding")
    assert after.metadata_location == before.metadata_location
    assert after.metadata == before.metadata


def test_failed_refresh_preserves_the_published_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    before = load_table_index(session, vectors, "vectors_embedding")

    with pytest.raises(parqdb.ParqDBError):
        vectors.refresh_index(
            "vectors_embedding",
            config=parqdb.IVF(nlist=3),
            wait_timeout=WAIT,
        )

    after = load_table_index(session, vectors, "vectors_embedding")
    status = vectors.index_status("vectors_embedding")
    assert after.metadata_location == before.metadata_location
    assert after.metadata == before.metadata
    assert status.state == "ready"
    assert status.current_snapshot_id == before.metadata["current-snapshot-id"]
    assert status.error is not None
    assert session.collect(vectors.search([0.0, 0.0]).limit(1))["id"].to_pylist() == [0]

    reopened = parqdb.connect(tmp_path / "parqdb-data")
    reopened_vectors = reopened.table("vectors")
    assert isinstance(reopened_vectors, parqdb.SourceTable)
    reopened_vectors.refresh_index("vectors_embedding", wait_timeout=WAIT)

    refreshed = vectors.index_status("vectors_embedding")
    assert refreshed.state == "ready"
    assert refreshed.current_snapshot_id != before.metadata["current-snapshot-id"]
    assert refreshed.error is None


def test_refresh_requires_an_index_for_the_bound_source(tmp_path: Path) -> None:
    first_source = tmp_path / "first.parquet"
    second_source = tmp_path / "second.parquet"
    write_vectors(first_source, [0], [[0.0, 0.0]])
    write_vectors(second_source, [0], [[0.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    first = register_source(session, first_source, "first")
    second = register_source(session, second_source, "second")
    build_index(first, nlist=1)

    with pytest.raises(parqdb.IndexNotFoundError):
        second.refresh_index("vectors_embedding")
    with pytest.raises(parqdb.IndexNotFoundError):
        first.refresh_index("missing")


def test_refresh_validates_config_and_timeout(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)

    with pytest.raises(TypeError, match=r"parqdb\.IVF"):
        vectors.refresh_index("missing", config=cast(Any, object()))
    with pytest.raises(ValueError, match="wait_timeout must be positive"):
        vectors.refresh_index("missing", wait_timeout=timedelta(0))

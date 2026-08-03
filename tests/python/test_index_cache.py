from __future__ import annotations

from pathlib import Path

import pytest
import relify
from _support import WAIT, build_index, register_source, write_vectors


def test_session_caches_complete_index_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors)

    cached = session.cache_index("vectors_embedding")

    assert cached.name == "vectors_embedding"
    assert (
        cached.snapshot_id
        == vectors.index_status("vectors_embedding").current_snapshot_id
    )
    assert cached.relation_count == 2
    assert cached.resident_bytes > 0
    assert session.is_index_cached("vectors_embedding")
    assert session.cache_index("vectors_embedding") == cached
    query = vectors.search([0.0, 0.0]).nprobes(1).limit(2).select(["id"])
    plan = session.explain(query)
    assert "CachedIvfScanExec" in plan
    assert "file_type=parquet" not in plan
    assert "FilterExec" not in plan
    assert "cid IN" not in plan
    assert session.to_arrow(query)["id"].to_pylist() == [0, 1]
    payload_query = vectors.search([0.0, 0.0]).nprobes(1).limit(2).select(["payload"])
    payload_plan = session.explain(payload_query)
    assert "HashJoinExec" in payload_plan
    assert "CachedIvfScanExec" in payload_plan
    assert "runtime_filters=1" in payload_plan
    assert session.to_arrow(payload_query)["payload"].to_pylist() == ["row-0", "row-1"]

    assert session.uncache_index("vectors_embedding")
    assert not session.uncache_index("vectors_embedding")
    assert not session.is_index_cached("vectors_embedding")


def test_refresh_invalidates_session_cache(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)
    session.cache_index("vectors_embedding")

    write_vectors(
        source,
        [0, 1, 2],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],
    )
    vectors.refresh_index("vectors_embedding", wait_timeout=WAIT)

    assert not session.is_index_cached("vectors_embedding")
    assert session.to_arrow(vectors.search([10.0, 0.0]).limit(1))["id"].to_pylist() == [
        2
    ]


def test_session_cache_validates_index_names(tmp_path: Path) -> None:
    session = relify.connect(tmp_path / "relify-data")

    with pytest.raises(TypeError, match="index name must be a string"):
        session.cache_index(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index name must not be empty"):
        session.cache_index("")
    with pytest.raises(relify.IndexNotFoundError):
        session.cache_index("missing")
    assert not session.is_index_cached("missing")

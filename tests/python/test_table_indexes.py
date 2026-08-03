from __future__ import annotations

from pathlib import Path

import pytest
import relify
from _support import build_index, register_source, write_vectors


def test_table_lists_only_indexes_for_its_exact_source(tmp_path: Path) -> None:
    first_source = tmp_path / "first.parquet"
    second_source = tmp_path / "second.parquet"
    write_vectors(first_source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    write_vectors(second_source, [2, 3], [[2.0, 0.0], [3.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    first = register_source(session, first_source, "first")
    second = register_source(session, second_source, "second")
    build_index(first, "first_embedding", nlist=1)
    build_index(second, "second_embedding", nlist=1)

    assert first.list_indexes() == [
        relify.IndexInfo(
            name="first_embedding",
            column="embedding",
            family="ivf",
            metric="l2_squared",
            parameters={
                "dimension": "2",
                "nlist": "1",
                "ntotal": "2",
                "store_vectors": "true",
            },
            current_snapshot_id=first.index_status(
                "first_embedding"
            ).current_snapshot_id,
        )
    ]
    assert [index.name for index in second.list_indexes()] == ["second_embedding"]
    with pytest.raises(TypeError):
        first.list_indexes()[0].parameters["nlist"] = "2"  # type: ignore[index]


def test_table_drop_is_source_scoped_and_preserves_catalog_consistency(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.parquet"
    second_source = tmp_path / "second.parquet"
    write_vectors(first_source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    write_vectors(second_source, [2, 3], [[2.0, 0.0], [3.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    first = register_source(session, first_source, "first")
    second = register_source(session, second_source, "second")
    build_index(first, "first_embedding", nlist=1)
    build_index(second, "second_embedding", nlist=1)

    with pytest.raises(relify.IndexNotFoundError):
        first.drop_index("second_embedding")
    assert session.indexes.list() == ["first_embedding", "second_embedding"]

    first.drop_index("first_embedding")

    assert first.list_indexes() == []
    assert [index.name for index in second.list_indexes()] == ["second_embedding"]
    assert session.indexes.list() == ["second_embedding"]

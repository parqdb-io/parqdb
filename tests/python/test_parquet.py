from __future__ import annotations

from pathlib import Path

import parqdb
import pyarrow.parquet as pq
import pytest
from _support import (
    build_index,
    load_table_index,
    register_source,
    relation_files,
    write_vectors,
)


@pytest.mark.parametrize(
    "reference_kind",
    ["file-path", "file-uri", "directory-path", "directory-uri"],
)
def test_parquet_source_reference_forms(
    tmp_path: Path,
    reference_kind: str,
) -> None:
    if reference_kind.startswith("file"):
        source = tmp_path / "vectors.parquet"
        write_vectors(
            source,
            [0, 1, 2, 3],
            [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
        )
    else:
        source = tmp_path / "vectors"
        source.mkdir()
        write_vectors(
            source / "part-0.parquet",
            [0, 1],
            [[0.0, 0.0], [1.0, 0.0]],
        )
        write_vectors(
            source / "part-1.parquet",
            [2, 3],
            [[10.0, 0.0], [11.0, 0.0]],
        )
    reference: str | Path = source
    if reference_kind.endswith("uri"):
        reference = source.as_uri()
        if source.is_dir():
            reference += "/"

    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, reference)
    build_index(vectors)

    hits = session.to_arrow(vectors.search([10.0, 0.0]).nprobes(2).limit(2))
    assert hits["id"].to_pylist() == [2, 3]
    snapshot = load_table_index(session, vectors, "vectors_embedding").metadata[
        "snapshots"
    ][0]
    source_uri = snapshot["source"]["uri"]
    assert source_uri.startswith("file://")
    assert source_uri.endswith("/") == reference_kind.startswith("directory")


def test_zstd_writer_options_reach_every_index_relation(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(
        vectors,
        writer_options=parqdb.WriteOptions(
            max_row_group_rows=1,
            write_batch_rows=1,
            partitions=2,
            compression="zstd(3)",
            target_file_size=64,
        ),
    )

    snapshot = load_table_index(session, vectors, "vectors_embedding").metadata[
        "snapshots"
    ][0]
    for role, reference in snapshot["index-relations"].items():
        paths = relation_files(reference)
        if role == "ivf_postings":
            assert len(paths) == 2
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                row_group = parquet_file.metadata.row_group(row_group_index)
                assert row_group.num_rows <= 1
                for column_index in range(row_group.num_columns):
                    assert row_group.column(column_index).compression == "ZSTD"


def test_postings_row_groups_default_to_bounded_average_cluster_size(
    tmp_path: Path,
) -> None:
    row_count = 8_193
    source = tmp_path / "vectors.parquet"
    write_vectors(
        source,
        list(range(row_count)),
        [[float(row % 97), float(row % 89)] for row in range(row_count)],
    )
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=2)

    snapshot = load_table_index(session, vectors, "vectors_embedding").metadata[
        "snapshots"
    ][0]
    postings = snapshot["index-relations"]["ivf_postings"]
    row_groups = [
        parquet_file.metadata.row_group(row_group)
        for path in relation_files(postings)
        for parquet_file in [pq.ParquetFile(path)]
        for row_group in range(parquet_file.metadata.num_row_groups)
    ]
    assert sum(row_group.num_rows for row_group in row_groups) == row_count
    assert len(row_groups) >= 2
    assert max(row_group.num_rows for row_group in row_groups) <= 8_192


def test_postings_use_one_hive_partitioned_file_per_cluster(tmp_path: Path) -> None:
    row_count = 512
    source = tmp_path / "vectors"
    source.mkdir()
    for part in range(8):
        start = part * row_count // 8
        end = (part + 1) * row_count // 8
        write_vectors(
            source / f"part-{part}.parquet",
            list(range(start, end)),
            [[float(row % 32), float(row // 32)] for row in range(start, end)],
        )
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    build_index(
        vectors,
        nlist=16,
        writer_options=parqdb.WriteOptions(
            partitions=4,
            target_file_size=1024 * 1024,
            max_row_group_rows=64,
            write_batch_rows=32,
        ),
    )

    snapshot = load_table_index(session, vectors, "vectors_embedding").metadata[
        "snapshots"
    ][0]
    postings = snapshot["index-relations"]["ivf_postings"]
    paths = relation_files(postings)
    assert len(paths) == 16
    assert sum(pq.ParquetFile(path).metadata.num_rows for path in paths) == row_count
    assert {path.name for path in paths} == {"part-00000.parquet"}
    assert {path.parent.name for path in paths} == {f"cid={cid}" for cid in range(16)}
    assert all(pq.ParquetFile(path).schema_arrow.names == ["key_1"] for path in paths)

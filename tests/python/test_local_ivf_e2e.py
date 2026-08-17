from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import parqdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from _support import (
    WAIT,
    load_metadata_file,
    load_table_index,
    register_source,
    relation_files,
    relation_path,
    thaw_json,
    vector_type,
)


def document_table(dimension: int = 2) -> pa.Table:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    required_vector = vector_type()
    schema = pa.schema(
        [
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=False),
            pa.field("embedding", required_vector, nullable=False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array(["a", "b", "c", "d"], type=pa.string()),
            pa.array(["zero", "one", "ten", "eleven"], type=pa.string()),
            pa.array(
                [
                    [value, *([0.0] * (dimension - 1))]
                    for value in [0.0, 1.0, 10.0, 11.0]
                ],
                type=required_vector,
            ),
        ],
        schema=schema,
    )


def write_documents(path: Path, dimension: int = 2) -> None:
    pq.write_table(document_table(dimension), path)


def write_partitioned_documents(root: Path) -> None:
    table = document_table()
    for partition, offset in [("p0", 0), ("p1", 2)]:
        directory = root / partition / "data"
        directory.mkdir(parents=True)
        pq.write_table(table.slice(offset, 2), directory / "part-0.parquet")


def test_local_build_publish_and_search(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_documents(source)

    session = parqdb.connect(tmp_path / "parqdb-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["document_id"],
        config=parqdb.IVF(nlist=2),
        writer_options=parqdb.WriteOptions(
            max_row_group_rows=2,
            write_batch_rows=1,
            partitions=2,
            compression="uncompressed",
            target_file_size=64,
        ),
        wait_timeout=WAIT,
    )

    status = documents.index_status("documents_embedding")
    assert status.state == "ready"
    assert status.current_snapshot_id is not None

    hits = session.to_arrow(
        documents.search([0.0, 0.0])
        .nprobes(1)
        .limit(2)
        .select(["document_id", "title"])
    )
    assert hits.column_names == ["document_id", "title", "_distance"]
    assert hits["document_id"].to_pylist() == ["a", "b"]
    assert hits["_distance"].to_pylist() == [0.0, 1.0]
    index_only_plan = session.explain(
        documents.search([0.0, 0.0]).nprobes(1).select(["document_id"])
    )
    payload_plan = session.explain(
        documents.search([0.0, 0.0]).nprobes(1).select(["title"])
    )
    assert "HashJoinExec" in index_only_plan
    assert "IvfTopKExec" in index_only_plan
    assert "HashJoinExec" in payload_plan
    assert "IvfTopKExec" in payload_plan
    vector_hits = session.to_arrow(
        documents.search([0.0, 0.0])
        .nprobes(1)
        .select(["document_id", "embedding"])
        .limit(2)
    )
    assert vector_hits["document_id"].to_pylist() == ["a", "b"]
    assert vector_hits["embedding"].to_pylist() == [[0.0, 0.0], [1.0, 0.0]]
    assert "HashJoinExec" in session.explain(
        documents.search([0.0, 0.0]).nprobes(1).select(["document_id", "embedding"])
    )

    exact = session.to_arrow(documents.search([0.0, 0.0]).nprobes(2).limit(4))
    assert exact["document_id"].to_pylist() == ["a", "b", "c", "d"]
    assert exact["_distance"].to_pylist() == [0.0, 1.0, 100.0, 121.0]

    tied = session.to_arrow(documents.search([0.5, 0.0]).nprobes(2).limit(2))
    assert set(tied["document_id"].to_pylist()) == {"a", "b"}
    assert tied["_distance"].to_pylist() == [0.25, 0.25]

    entry = load_table_index(session, documents, "documents_embedding")
    assert entry.metadata["format-version"] == 1
    snapshot = entry.metadata["snapshots"][0]
    assert snapshot["index-family"] == "ivf"
    assert snapshot["parameters"]["dimension"] == "2"
    assert snapshot["parameters"]["nlist"] == "2"
    assert snapshot["parameters"]["ntotal"] == "4"
    assert snapshot["parameters"]["posting_encoding"] == "source"
    assert snapshot["parameters"]["ivf_centroids_fingerprint"]
    assert snapshot["parameters"]["ivf_centroids_uuid"]
    assert snapshot["parameters"]["ivf_centroids_metadata_location"]
    assert set(snapshot["index-relations"]) == {
        "ivf_centroids",
        "ivf_postings",
    }
    required_vector = vector_type()
    assert pq.read_schema(
        relation_path(snapshot["index-relations"]["ivf_centroids"])
    ) == pa.schema(
        [
            pa.field("cid", pa.int32(), nullable=False),
            pa.field("centroid", required_vector, nullable=False),
        ]
    )
    assert pq.read_schema(
        relation_path(snapshot["index-relations"]["ivf_postings"])
    ) == pa.schema(
        [
            pa.field("key_1", pa.string(), nullable=False),
        ]
    )
    posting_files = relation_files(snapshot["index-relations"]["ivf_postings"])
    assert posting_files
    for file in posting_files:
        parquet_file = pq.ParquetFile(file)
        for row_group in range(parquet_file.metadata.num_row_groups):
            metadata = parquet_file.metadata.row_group(row_group)
            assert metadata.num_rows <= 2
            assert metadata.column(0).compression == "UNCOMPRESSED"
            assert metadata.column(0).statistics is not None
        assert file.parent.name.startswith("cid=")
        assert "cid" not in parquet_file.schema_arrow.names
    assert load_metadata_file(entry.metadata_location) == thaw_json(entry.metadata)

    reopened = parqdb.connect(tmp_path / "parqdb-data")
    documents = reopened.table("documents")
    assert isinstance(documents, parqdb.SourceTable)
    hits = reopened.to_arrow(documents.search([10.0, 0.0]).nprobes(1).limit(2))
    assert hits["document_id"].to_pylist() == ["c", "d"]


def test_local_lvq_build_and_search(tmp_path: Path) -> None:
    dimension = 32
    query_vector = [10.0, *([0.0] * (dimension - 1))]
    source = tmp_path / "documents.parquet"
    write_documents(source, dimension)
    session = parqdb.connect(tmp_path / "parqdb-data")
    documents = register_source(session, source, "documents")

    for encoding in ["lvq4", "lvq8"]:
        name = f"documents_{encoding}"
        documents.create_index(
            name,
            column="embedding",
            key=["document_id"],
            config=parqdb.IVF(nlist=2, encoding=encoding),
            wait_timeout=WAIT,
        )
        snapshot = load_table_index(session, documents, name).metadata["snapshots"][0]
        assert snapshot["index-schema-version"] == 1
        assert snapshot["parameters"]["posting_encoding"] == encoding
        assert snapshot["parameters"]["dimension"] == str(dimension)
        postings = snapshot["index-relations"]["ivf_postings"]
        posting_files = relation_files(postings)
        assert posting_files
        assert pq.read_schema(posting_files[0]) == pa.schema(
            [
                pa.field("key_1", pa.string(), nullable=False),
                pa.field("offset", pa.float32(), nullable=False),
                pa.field("scale", pa.float32(), nullable=False),
                pa.field("code", pa.binary_view(), nullable=False),
            ]
        )
        for file in posting_files:
            parquet_file = pq.ParquetFile(file)
            code_index = parquet_file.schema_arrow.get_field_index("code")
            for row_group in range(parquet_file.metadata.num_row_groups):
                code_chunk = parquet_file.metadata.row_group(row_group).column(
                    code_index
                )
                assert code_chunk.physical_type == "BYTE_ARRAY"
                assert "PLAIN" in code_chunk.encodings
                assert "RLE_DICTIONARY" not in code_chunk.encodings
        expected_code_size = dimension if encoding == "lvq8" else dimension // 2
        codes = [
            code
            for file in posting_files
            for code in pq.read_table(file)["code"].to_pylist()
        ]
        assert codes
        assert expected_code_size > 12
        assert all(len(code) == expected_code_size for code in codes)

        query = (
            documents.search(query_vector, index=name)
            .nprobes(2)
            .select(["document_id"])
            .limit(1)
        )
        hits = session.to_arrow(query)
        assert hits["document_id"].to_pylist() == ["c"]
        plan = session.explain(query)
        assert "IvfTopKExec" in plan
        assert "HashJoinExec" not in plan

        payload_query = (
            documents.search(query_vector, index=name)
            .nprobes(2)
            .select(["document_id", "title"])
            .limit(1)
        )
        payload = session.to_arrow(payload_query)
        assert payload["title"].to_pylist() == ["ten"]
        assert "HashJoinExec" in session.explain(payload_query)


@pytest.mark.skipif(sys.platform != "linux", reason="Direct I/O requires Linux")
def test_local_lvq_search_with_direct_io(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_documents(source, 32)
    config = (
        parqdb.SessionConfig()
        .set("parqdb.parquet.index_io", "direct")
        .set("parqdb.parquet.page_cache.capacity", str(8 * 1024 * 1024))
    )
    session = parqdb.connect(tmp_path / "parqdb-data", config=config)
    documents = register_source(session, source, "documents")
    documents.create_index(
        "documents_lvq8",
        column="embedding",
        key=["document_id"],
        config=parqdb.IVF(nlist=2, encoding="lvq8"),
        wait_timeout=WAIT,
    )
    query = (
        documents.search([10.0, *([0.0] * 31)], index="documents_lvq8")
        .nprobes(2)
        .select(["document_id"])
        .limit(1)
    )

    assert session.to_arrow(query)["document_id"].to_pylist() == ["c"]
    cold = session.parquet_page_cache_stats()
    assert cold.admissions > 0
    assert session.to_arrow(query)["document_id"].to_pylist() == ["c"]
    assert session.parquet_page_cache_stats().hits > cold.hits


def test_ivf_centroids_float64_and_cosine_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "cosine.parquet"
    vector = pa.list_(pa.field("element", pa.float64(), nullable=False))
    table = pa.Table.from_arrays(
        [
            pa.array(["x", "diagonal", "y"], type=pa.string()),
            pa.array([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]], type=vector),
        ],
        schema=pa.schema(
            [
                pa.field("document_id", pa.string(), nullable=False),
                pa.field("embedding", vector, nullable=False),
            ]
        ),
    )
    pq.write_table(table, source)
    session = parqdb.connect(tmp_path / "parqdb-data")
    documents = register_source(session, source, "documents")

    cosine_snapshots = []
    for encoding in ("source", "lvq4", "lvq8"):
        name = f"cosine_{encoding}"
        documents.create_index(
            name,
            column="embedding",
            key=["document_id"],
            config=parqdb.IVF(nlist=1, encoding=encoding, metric="cosine"),
            wait_timeout=WAIT,
        )
        snapshot = load_table_index(session, documents, name).metadata["snapshots"][0]
        cosine_snapshots.append(snapshot)
        hits = session.to_arrow(
            documents.search([10.0, 0.0], index=name).nprobes(1).limit(3)
        )
        assert hits["document_id"].to_pylist() == ["x", "diagonal", "y"]
        assert hits["_distance"].to_pylist() == pytest.approx(
            [0.0, 1.0 - 2.0**-0.5, 1.0], abs=1e-5
        )

    centroid_fields = (
        "ivf_centroids_fingerprint",
        "ivf_centroids_uuid",
        "ivf_centroids_metadata_location",
    )
    for field in centroid_fields:
        assert (
            len({snapshot["parameters"][field] for snapshot in cosine_snapshots}) == 1
        )
    centroid_relations = [
        snapshot["index-relations"]["ivf_centroids"] for snapshot in cosine_snapshots
    ]
    assert all(relation == centroid_relations[0] for relation in centroid_relations[1:])
    assert (
        len(
            {
                snapshot["index-relations"]["ivf_postings"]["uri"]
                for snapshot in cosine_snapshots
            }
        )
        == 3
    )

    documents.create_index(
        "l2_source",
        column="embedding",
        key=["document_id"],
        config=parqdb.IVF(nlist=1, metric="l2_squared"),
        wait_timeout=WAIT,
    )
    l2_snapshot = load_table_index(session, documents, "l2_source").metadata[
        "snapshots"
    ][0]
    assert (
        l2_snapshot["parameters"]["ivf_centroids_fingerprint"]
        != cosine_snapshots[0]["parameters"]["ivf_centroids_fingerprint"]
    )
    l2_hits = session.to_arrow(
        documents.search([2.0, 0.0], index="l2_source").nprobes(1).limit(3)
    )
    assert l2_hits["document_id"].to_pylist() == ["x", "diagonal", "y"]
    assert l2_hits["_distance"].to_pylist() == pytest.approx([0.0, 2.0, 13.0])

    with pytest.raises(parqdb.InvalidArgumentError, match="non-zero norm"):
        session.to_arrow(documents.search([0.0, 0.0], index="cosine_source"))


def test_cosine_build_rejects_zero_source_vectors(tmp_path: Path) -> None:
    source = tmp_path / "zero.parquet"
    required_vector = vector_type()
    pq.write_table(
        pa.table(
            {
                "document_id": pa.array(["zero"], type=pa.string()),
                "embedding": pa.array([[0.0, 0.0]], type=required_vector),
            },
            schema=pa.schema(
                [
                    pa.field("document_id", pa.string(), nullable=False),
                    pa.field("embedding", required_vector, nullable=False),
                ]
            ),
        ),
        source,
    )
    session = parqdb.connect(tmp_path / "parqdb-data")
    documents = register_source(session, source, "documents")

    with pytest.raises(parqdb.InvalidSchemaError, match="non-zero norm"):
        documents.create_index(
            "cosine_index",
            column="embedding",
            key=["document_id"],
            config=parqdb.IVF(nlist=1, metric="cosine"),
            wait_timeout=WAIT,
        )


def test_index_and_source_catalog_survive_python_process_restart(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "documents"
    source = source_root / "*" / "data" / "*.parquet"
    root = tmp_path / "parqdb-data"
    write_partitioned_documents(source_root)
    build = """
import parqdb
import sys

session = parqdb.connect(sys.argv[1])
session.register_parquet("documents", sys.argv[2])
documents = session.table("documents")
documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=parqdb.IVF(nlist=2),
)
documents.wait_for_index("documents_embedding")
"""
    query = """
import json
import parqdb
import sys

session = parqdb.connect(sys.argv[1])
documents = session.table("documents")
query = (
    documents.search([10.0, 0.0])
    .nprobes(1)
    .limit(2)
    .select(["document_id", "title"])
)
hits = session.to_arrow(query)
metadata = session._indexes.load(
    "documents_embedding",
    namespace=documents.identifier.index_namespace,
).metadata
print(json.dumps({
    "hits": hits.to_pydict(),
    "source": metadata["snapshots"][0]["source"]["uri"],
}, sort_keys=True))
"""

    subprocess.run(
        [sys.executable, "-c", build, str(root), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        [sys.executable, "-c", query, str(root)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["hits"] == {
        "_distance": [0.0, 1.0],
        "document_id": ["c", "d"],
        "title": ["ten", "eleven"],
    }
    assert payload["source"].startswith("file://")
    assert payload["source"].endswith("/documents/*/data/*.parquet")


def test_composite_keys_are_stored_directly_in_postings(tmp_path: Path) -> None:
    source = tmp_path / "composite.parquet"
    required_vector = vector_type()
    pq.write_table(
        pa.table(
            {
                "tenant": pa.array(["b", "a", "b", "a"], type=pa.string()),
                "document_id": pa.array([2, 2, 1, 1], type=pa.int64()),
                "embedding": pa.array(
                    [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    type=required_vector,
                ),
            },
            schema=pa.schema(
                [
                    pa.field("tenant", pa.string(), nullable=False),
                    pa.field("document_id", pa.int64(), nullable=False),
                    pa.field("embedding", required_vector, nullable=False),
                ]
            ),
        ),
        source,
    )
    session = parqdb.connect(tmp_path / "parqdb-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "composite_index",
        column="embedding",
        key=["tenant", "document_id"],
        config=parqdb.IVF(nlist=2),
        wait_timeout=WAIT,
    )

    hits = session.to_arrow(documents.search([0.5, 0.0]).nprobes(2).limit(4))
    assert set(zip(hits["tenant"].to_pylist(), hits["document_id"].to_pylist())) == {
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    }
    assert hits["_distance"].to_pylist() == [0.25, 0.25, 0.25, 0.25]
    snapshot = load_table_index(session, documents, "composite_index").metadata[
        "snapshots"
    ][0]
    assert set(snapshot["index-relations"]) == {"ivf_centroids", "ivf_postings"}
    assert pq.read_schema(
        relation_path(snapshot["index-relations"]["ivf_postings"])
    ) == pa.schema(
        [
            pa.field("key_1", pa.string(), nullable=False),
            pa.field("key_2", pa.int64(), nullable=False),
        ]
    )


def test_vectors_can_be_omitted_from_postings(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_documents(source)
    session = parqdb.connect(tmp_path / "parqdb-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "compact_index",
        column="embedding",
        key=["document_id"],
        config=parqdb.IVF(nlist=2, encoding="source"),
        wait_timeout=WAIT,
    )

    snapshot = load_table_index(session, documents, "compact_index").metadata[
        "snapshots"
    ][0]
    assert snapshot["parameters"]["posting_encoding"] == "source"
    assert pq.read_schema(
        relation_path(snapshot["index-relations"]["ivf_postings"])
    ) == pa.schema(
        [
            pa.field("key_1", pa.string(), nullable=False),
        ]
    )
    hits = session.to_arrow(
        documents.search([0.0, 0.0], index="compact_index")
        .nprobes(1)
        .select(["document_id"])
        .limit(2)
    )
    assert hits["document_id"].to_pylist() == ["a", "b"]
    assert "HashJoinExec" in session.explain(
        documents.search([0.0, 0.0], index="compact_index")
        .nprobes(1)
        .select(["document_id"])
    )


def test_gemm_build_and_simd_search_path(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    dimension = 8
    required_vector = vector_type()
    pq.write_table(
        pa.table(
            {
                "id": pa.array(range(32), type=pa.int64()),
                "embedding": pa.array(
                    [[float(row)] * dimension for row in range(32)],
                    type=required_vector,
                ),
            },
            schema=pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    pa.field("embedding", required_vector, nullable=False),
                ]
            ),
        ),
        source,
    )
    session = parqdb.connect(tmp_path / "parqdb-data")
    vectors = register_source(session, source)
    vectors.create_index(
        "gemm_index",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=16),
        wait_timeout=WAIT,
    )

    hits = session.to_arrow(
        vectors.search([7.0] * dimension).nprobes(16).limit(1).select(["id"])
    )
    assert hits["id"].to_pylist() == [7]
    assert hits["_distance"].to_pylist() == [0.0]

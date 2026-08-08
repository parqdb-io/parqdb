from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import relify
from _support import (
    WAIT,
    load_metadata_file,
    register_source,
    relation_files,
    relation_path,
    thaw_json,
    vector_type,
)
from relify.backends.v1 import QueryProfile
from relify.testing import BackendQueryCase, check_query_backend


def document_table() -> pa.Table:
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
                [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
                type=required_vector,
            ),
        ],
        schema=schema,
    )


def write_documents(path: Path) -> None:
    pq.write_table(document_table(), path)


def write_partitioned_documents(root: Path) -> None:
    table = document_table()
    for partition, offset in [("p0", 0), ("p1", 2)]:
        directory = root / partition / "data"
        directory.mkdir(parents=True)
        pq.write_table(table.slice(offset, 2), directory / "part-0.parquet")


def test_local_build_publish_and_search(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_documents(source)

    session = relify.connect(tmp_path / "relify-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["document_id"],
        config=relify.IVF(nlist=2),
        builder=relify.Local(
            max_row_group_rows=2,
            write_batch_rows=1,
        ),
        writer_options=relify.WriteOptions(
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
    contract_query = (
        documents.search([0.0, 0.0]).nprobes(1).limit(2).select(["document_id"])
    )
    contract_schema = pa.schema(
        [
            pa.field("document_id", pa.string(), nullable=False),
            pa.field("_distance", pa.float32(), nullable=False),
        ]
    )
    check_query_backend(
        session,
        [
            BackendQueryCase(
                "local parquet IVF",
                QueryProfile("ivf", "parquet", "parquet"),
                contract_query,
                pa.Table.from_arrays(
                    [
                        pa.array(["a", "b"], type=pa.string()),
                        pa.array([0.0, 1.0], type=pa.float32()),
                    ],
                    schema=contract_schema,
                ),
            )
        ],
    )
    index_only_plan = session.explain(
        documents.search([0.0, 0.0]).nprobes(1).select(["document_id"])
    )
    payload_plan = session.explain(
        documents.search([0.0, 0.0]).nprobes(1).select(["title"])
    )
    assert "HashJoinExec" not in index_only_plan
    assert "IvfTopKExec" in index_only_plan
    assert "ProjectionExec" not in index_only_plan
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
    assert "HashJoinExec" not in session.explain(
        documents.search([0.0, 0.0]).nprobes(1).select(["document_id", "embedding"])
    )

    exact = session.to_arrow(documents.search([0.0, 0.0]).nprobes(2).limit(4))
    assert exact["document_id"].to_pylist() == ["a", "b", "c", "d"]
    assert exact["_distance"].to_pylist() == [0.0, 1.0, 100.0, 121.0]

    tied = session.to_arrow(documents.search([0.5, 0.0]).nprobes(2).limit(2))
    assert set(tied["document_id"].to_pylist()) == {"a", "b"}
    assert tied["_distance"].to_pylist() == [0.25, 0.25]

    entry = session.indexes.load("documents_embedding")
    assert entry.metadata["format-version"] == 1
    snapshot = entry.metadata["snapshots"][0]
    assert snapshot["index-family"] == "ivf"
    assert snapshot["parameters"] == {
        "dimension": "2",
        "nlist": "2",
        "ntotal": "4",
        "store_vectors": "true",
    }
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
            pa.field("vector", required_vector, nullable=False),
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

    reopened = relify.connect(tmp_path / "relify-data")
    documents = reopened.table("documents")
    assert isinstance(documents, relify.SourceTable)
    hits = reopened.to_arrow(documents.search([10.0, 0.0]).nprobes(1).limit(2))
    assert hits["document_id"].to_pylist() == ["c", "d"]


def test_local_lvq_build_and_search(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_documents(source)
    session = relify.connect(tmp_path / "relify-data")
    documents = register_source(session, source, "documents")

    for encoding, code_size in [("lvq4", 1), ("lvq8", 2)]:
        name = f"documents_{encoding}"
        documents.create_index(
            name,
            column="embedding",
            key=["document_id"],
            config=relify.IVF(nlist=2, posting_encoding=encoding),
            wait_timeout=WAIT,
        )
        snapshot = session.indexes.load(name).metadata["snapshots"][0]
        assert snapshot["index-schema-version"] == 2
        assert snapshot["parameters"]["posting_encoding"] == encoding
        assert pq.read_schema(
            relation_path(snapshot["index-relations"]["ivf_postings"])
        ) == pa.schema(
            [
                pa.field("key_1", pa.string(), nullable=False),
                pa.field("offset", pa.float32(), nullable=False),
                pa.field("scale", pa.float32(), nullable=False),
                pa.field("code", pa.binary(code_size), nullable=False),
            ]
        )

        query = (
            documents.search([10.0, 0.0], index=name)
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
            documents.search([10.0, 0.0], index=name)
            .nprobes(2)
            .select(["document_id", "title"])
            .limit(1)
        )
        payload = session.to_arrow(payload_query)
        assert payload["title"].to_pylist() == ["ten"]
        assert "HashJoinExec" in session.explain(payload_query)


def test_index_and_source_catalog_survive_python_process_restart(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "documents"
    source = source_root / "*" / "data" / "*.parquet"
    root = tmp_path / "relify-data"
    write_partitioned_documents(source_root)
    build = """
import relify
import sys

session = relify.connect(sys.argv[1])
session.register_parquet("documents", sys.argv[2])
documents = session.table("documents")
documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=2),
)
documents.wait_for_index("documents_embedding")
"""
    query = """
import json
import relify
import sys

session = relify.connect(sys.argv[1])
documents = session.table("documents")
query = (
    documents.search([10.0, 0.0])
    .nprobes(1)
    .limit(2)
    .select(["document_id", "title"])
)
hits = session.to_arrow(query)
metadata = session.indexes.load("documents_embedding").metadata
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
    session = relify.connect(tmp_path / "relify-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "composite_index",
        column="embedding",
        key=["tenant", "document_id"],
        config=relify.IVF(nlist=2),
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
    snapshot = session.indexes.load("composite_index").metadata["snapshots"][0]
    assert set(snapshot["index-relations"]) == {"ivf_centroids", "ivf_postings"}
    assert pq.read_schema(
        relation_path(snapshot["index-relations"]["ivf_postings"])
    ) == pa.schema(
        [
            pa.field("key_1", pa.string(), nullable=False),
            pa.field("key_2", pa.int64(), nullable=False),
            pa.field("vector", required_vector, nullable=False),
        ]
    )


def test_vectors_can_be_omitted_from_postings(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_documents(source)
    session = relify.connect(tmp_path / "relify-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "compact_index",
        column="embedding",
        key=["document_id"],
        config=relify.IVF(nlist=2, store_vectors=False),
        wait_timeout=WAIT,
    )

    snapshot = session.indexes.load("compact_index").metadata["snapshots"][0]
    assert snapshot["parameters"]["store_vectors"] == "false"
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
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    vectors.create_index(
        "gemm_index",
        column="embedding",
        key=["id"],
        config=relify.IVF(nlist=16),
        wait_timeout=WAIT,
    )

    hits = session.to_arrow(
        vectors.search([7.0] * dimension).nprobes(16).limit(1).select(["id"])
    )
    assert hits["id"].to_pylist() == [7]
    assert hits["_distance"].to_pylist() == [0.0]

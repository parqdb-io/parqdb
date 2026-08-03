from __future__ import annotations

import pytest
import relify
from relify.backends.v1 import (
    ResolvedSearch,
    resolve_indexed_search,
    resolve_projection,
    resolve_vector_field,
)


def test_resolved_search_contains_only_portable_backend_input() -> None:
    query = (
        relify.VectorQuery(
            source=relify.TableIdentifier("lakehouse", ("analytics",), "documents"),
            query=(0.0, 1.0),
            column="embedding",
        )
        .where("tenant_id = 42")
        .nprobes(1)
        .limit(10)
        .select(["document_id"])
    )

    search = resolve_indexed_search(
        query,
        index="documents_embedding",
        metadata=_metadata(),
        projection=("document_id",),
    )

    assert isinstance(search, ResolvedSearch)
    assert search.index == "documents_embedding"
    assert search.query_vector == (0.0, 1.0)
    assert search.nlist == 2
    assert search.nprobe == 1
    assert search.source_key_fields == ("document_id",)
    assert search.store_vectors
    assert search.needs_source
    assert search.source_relation["snapshot-id"] == 101
    assert search.source_relation["namespace"] == ("analytics",)
    assert search.centroids_relation["snapshot-id"] == 201
    assert search.postings_relation["snapshot-id"] == 202
    with pytest.raises(TypeError):
        search.source_relation["snapshot-id"] = 102  # type: ignore[index]


def test_index_only_resolution_is_shared_by_every_backend() -> None:
    query = relify.VectorQuery(
        source=relify.TableIdentifier("lakehouse", ("analytics",), "documents"),
        query=(0.0, 1.0),
        column="embedding",
        projection=("document_id",),
    )

    search = resolve_indexed_search(
        query,
        index="documents_embedding",
        metadata=_metadata(),
        projection=("document_id",),
    )

    assert not search.needs_source


def test_shared_projection_and_vector_resolution_report_schema_errors() -> None:
    assert resolve_projection(("id", "embedding"), None) == ("id", "embedding")
    with pytest.raises(ValueError, match="unknown source fields"):
        resolve_projection(("id", "embedding"), ("missing",))
    with pytest.raises(TypeError, match=r"list<float>"):
        resolve_vector_field(
            requested="id",
            source_fields=("id", "embedding"),
            vector_fields=("embedding",),
        )


def _metadata() -> dict[str, object]:
    source = {
        "profile": "iceberg",
        "catalog": "lakehouse",
        "namespace": ["analytics"],
        "name": "documents",
        "table-uuid": "11111111-1111-1111-1111-111111111111",
        "snapshot-id": 101,
    }
    return {
        "format-version": 1,
        "index-uuid": "22222222-2222-2222-2222-222222222222",
        "location": "file:///tmp/index",
        "current-snapshot-id": 1,
        "snapshots": [
            {
                "snapshot-id": 1,
                "sequence-number": 1,
                "timestamp-ms": 1,
                "index-family": "ivf",
                "index-schema-version": 1,
                "metric": "l2_squared",
                "source": source,
                "source-key-fields": ["document_id"],
                "vector-field": "embedding",
                "parameters": {
                    "dimension": "2",
                    "nlist": "2",
                    "ntotal": "3",
                    "store_vectors": "true",
                },
                "index-relations": {
                    "ivf_centroids": {
                        **source,
                        "namespace": ["relify"],
                        "name": "documents_embedding_centroids",
                        "snapshot-id": 201,
                    },
                    "ivf_postings": {
                        **source,
                        "namespace": ["relify"],
                        "name": "documents_embedding_postings",
                        "snapshot-id": 202,
                    },
                },
                "summary": {"operation": "create"},
            }
        ],
        "snapshot-log": [{"timestamp-ms": 1, "snapshot-id": 1}],
    }

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import relify
from pyiceberg.schema import Schema
from pyiceberg.types import (
    FloatType,
    IntegerType,
    ListType,
    LongType,
    NestedField,
    StringType,
)
from support.indexes import write_shared_ivf_metadata


@dataclass(frozen=True)
class _Snapshot:
    snapshot_id: int
    schema_id: int


@dataclass(frozen=True)
class _Metadata:
    table_uuid: uuid.UUID


class _IcebergTable:
    def __init__(self, snapshot_id: int, schema: Schema) -> None:
        self.metadata = _Metadata(uuid.uuid4())
        self._snapshot = _Snapshot(snapshot_id, schema.schema_id)
        self._schema = schema

    def current_snapshot(self) -> _Snapshot:
        return self._snapshot

    def snapshot_by_id(self, snapshot_id: int) -> _Snapshot | None:
        return self._snapshot if snapshot_id == self._snapshot.snapshot_id else None

    def schemas(self) -> dict[int, Schema]:
        return {self._schema.schema_id: self._schema}


class _IcebergCatalog:
    name = "lakehouse"

    def __init__(self) -> None:
        self.tables: dict[tuple[str, ...], _IcebergTable] = {}

    def load_table(self, identifier: tuple[str, ...]) -> _IcebergTable:
        return self.tables[identifier]


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self._sql = ""

    def execute(self, sql: str) -> None:
        self._sql = sql
        self._connection.executed.append(sql)

    def fetch_record_batch(self) -> pa.RecordBatchReader:
        batches = (
            []
            if self._sql.endswith("LIMIT 0")
            else list(self._connection.result_batches)
        )
        schema = pa.schema([]) if not batches else batches[0].schema
        return pa.RecordBatchReader.from_batches(schema, batches)

    def fetchall(self) -> list[tuple[str]]:
        return [("PLAN FRAGMENT 0",), ("  TOP-N",)]

    def close(self) -> None:
        self._connection.closed_cursors += 1


class _Connection:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.closed_cursors = 0
        self.result_batches: list[pa.RecordBatch] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)


def _field(
    field_id: int,
    name: str,
    field_type: Any,
    *,
    required: bool = True,
) -> NestedField:
    return NestedField(
        field_id=field_id,
        name=name,
        field_type=field_type,
        required=required,
    )


def _schemas() -> tuple[Schema, Schema, Schema]:
    vector = ListType(
        element_id=20,
        element=FloatType(),
        element_required=True,
    )
    source = Schema(
        _field(1, "document_id", LongType()),
        _field(2, "embedding", vector),
        _field(3, "tenant_id", IntegerType()),
        _field(4, "title", StringType()),
        schema_id=1,
    )
    centroids = Schema(
        _field(10, "cid", IntegerType()),
        _field(
            11,
            "centroid",
            ListType(
                element_id=12,
                element=FloatType(),
                element_required=True,
            ),
        ),
        schema_id=2,
    )
    postings = Schema(
        _field(30, "cid", IntegerType()),
        _field(31, "key_1", LongType()),
        schema_id=3,
    )
    return source, centroids, postings


def _relation(
    catalog: _IcebergCatalog,
    identifier: tuple[str, ...],
) -> dict[str, object]:
    table = catalog.load_table(identifier)
    return {
        "profile": "iceberg",
        "catalog": "lakehouse",
        "namespace": list(identifier[:-1]),
        "name": identifier[-1],
        "table-uuid": str(table.metadata.table_uuid),
        "snapshot-id": table.current_snapshot().snapshot_id,
    }


def _session(tmp_path: Path) -> tuple[Any, _Connection, _IcebergCatalog]:
    source_schema, centroid_schema, posting_schema = _schemas()
    catalog = _IcebergCatalog()
    catalog.tables = {
        ("analytics", "documents"): _IcebergTable(101, source_schema),
        ("relify", "documents_embedding_centroids"): _IcebergTable(
            201,
            centroid_schema,
        ),
        ("relify", "documents_embedding_postings"): _IcebergTable(
            202,
            posting_schema,
        ),
    }
    connection = _Connection()
    metadata_root = tmp_path / "metadata"
    session = relify.experimental.starrocks.connect(
        connection,
        index_catalog=f"sqlite://{tmp_path / 'catalog.sqlite'}",
        iceberg_catalog=catalog,
        metadata_root=metadata_root.as_uri(),
    )
    source = _relation(catalog, ("analytics", "documents"))
    centroids = _relation(catalog, ("relify", "documents_embedding_centroids"))
    shared = write_shared_ivf_metadata(
        metadata_root,
        source=source,
        centroids=centroids,
    )
    session._native.publish_initial(
        index_name="documents_embedding",
        source_json=json.dumps(source),
        vector_field="embedding",
        source_key_fields=["document_id"],
        builder="fixture",
        metric="l2_squared",
        parameters={
            "dimension": "2",
            "nlist": "2",
            "ntotal": "3",
            "posting_encoding": "source",
            **shared,
        },
        index_relations={
            "ivf_centroids": json.dumps(centroids),
            "ivf_postings": json.dumps(
                _relation(catalog, ("relify", "documents_embedding_postings"))
            ),
        },
    )
    return session, connection, catalog


def test_starrocks_compiles_and_collects_an_iceberg_vector_query(
    tmp_path: Path,
) -> None:
    session, connection, _ = _session(tmp_path)
    documents = session.table("analytics.documents")
    query = (
        documents.search([0.0, 1.0], column="embedding")
        .where("tenant_id = 42")
        .nprobes(1)
        .limit(10_000)
        .select(["document_id", "title"])
    )

    sql = session.to_sql(query)

    assert documents.identifier == relify.TableIdentifier(
        "lakehouse",
        ("analytics",),
        "documents",
    )
    assert (
        "FROM `lakehouse`.`relify`.`documents_embedding_centroids` "
        "FOR VERSION AS OF 201"
    ) in sql
    assert (
        "FROM `lakehouse`.`relify`.`documents_embedding_postings` "
        "FOR VERSION AS OF 202 AS p"
    ) in sql
    assert ("FROM `lakehouse`.`analytics`.`documents` FOR VERSION AS OF 101") in sql
    assert "ORDER BY l2_distance(ARRAY<FLOAT>[0.0, 1.0], `centroid`) ASC" in sql
    assert "WHERE (tenant_id = 42)" in sql
    assert "p.`key_1` = s.`document_id`" in sql
    assert "CAST(`__relify_squared_l2` AS FLOAT)" in sql
    assert "`__relify_squared_l2` * `__relify_squared_l2`" not in sql
    assert sql.endswith("LIMIT 10000")

    connection.result_batches = [
        pa.record_batch(
            [
                pa.array([1, 2], type=pa.int64()),
                pa.array(["a", "b"]),
                pa.array([0.0, 1.0], type=pa.float32()),
            ],
            names=["document_id", "title", "_distance"],
        )
    ]
    result = session.collect(query)

    assert result.to_pydict() == {
        "document_id": [1, 2],
        "title": ["a", "b"],
        "_distance": [0.0, 1.0],
    }
    assert connection.closed_cursors == 2


def test_starrocks_source_encoding_joins_the_source_for_scoring(
    tmp_path: Path,
) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")

    sql = session.to_sql(
        documents.search([0.0, 0.0], column="embedding")
        .nprobes(1)
        .select(["document_id"])
    )

    assert "`lakehouse`.`analytics`.`documents`" in sql
    assert "s.`document_id` AS `document_id`" in sql
    assert "INNER JOIN _relify_source" in sql


def test_starrocks_source_join_supplies_vectors_when_postings_omit_them(
    tmp_path: Path,
) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")

    sql = session.to_sql(
        documents.search([0.0, 0.0], column="embedding")
        .nprobes(2)
        .select(["document_id"])
    )

    assert "documents_embedding_centroids" not in sql
    assert "l2_distance(ARRAY<FLOAT>[0.0, 0.0], s.`embedding`)" in sql
    assert "INNER JOIN _relify_source AS s" in sql


def test_starrocks_exact_search_does_not_resolve_an_index(tmp_path: Path) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")

    sql = session.to_sql(
        documents.search([0.0, 0.0], column="embedding")
        .bypass_vector_index()
        .limit(1)
        .select(["document_id"])
    )

    assert "documents_embedding_postings" not in sql
    assert "`lakehouse`.`analytics`.`documents` FOR VERSION AS OF 101" in sql
    assert "l2_distance(ARRAY<FLOAT>[0.0, 0.0], s.`embedding`)" in sql


def test_starrocks_explain_uses_the_compiled_server_side_query(tmp_path: Path) -> None:
    session, connection, _ = _session(tmp_path)
    documents = session.table("analytics.documents")

    plan = session.explain(documents.search([0.0, 0.0], column="embedding"))

    assert plan == "PLAN FRAGMENT 0\n  TOP-N"
    assert connection.executed[-1].startswith("EXPLAIN WITH\n")


def test_starrocks_rejects_non_finite_or_wrongly_typed_distances(
    tmp_path: Path,
) -> None:
    session, connection, _ = _session(tmp_path)
    documents = session.table("analytics.documents")
    query = documents.search([0.0, 0.0], column="embedding").select(["document_id"])
    connection.result_batches = [
        pa.record_batch(
            [
                pa.array([1], type=pa.int64()),
                pa.array([float("inf")], type=pa.float32()),
            ],
            names=["document_id", "_distance"],
        )
    ]
    with pytest.raises(ValueError, match="non-finite"):
        session.collect(query)

    connection.result_batches = [
        pa.record_batch(
            [
                pa.array([1], type=pa.int64()),
                pa.array([0.0], type=pa.float64()),
            ],
            names=["document_id", "_distance"],
        )
    ]
    with pytest.raises(TypeError, match="float32"):
        session.collect(query)


def test_starrocks_has_no_implicit_builder(tmp_path: Path) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")

    assert session.default_builder is None
    with pytest.raises(ValueError, match="no default index builder"):
        documents.create_index(
            "other_embedding",
            column="embedding",
            key=["document_id"],
            config=relify.IVF(nlist=2),
        )
    assert documents.list_indexes()[0].name == "documents_embedding"


def test_starrocks_drop_reports_a_missing_index(tmp_path: Path) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")

    documents.drop_index("documents_embedding")
    with pytest.raises(relify.IndexNotFoundError, match="index not found"):
        documents.drop_index("documents_embedding")


def test_starrocks_table_builds_through_an_explicit_independent_builder(
    tmp_path: Path,
) -> None:
    session, _, catalog = _session(tmp_path)
    documents = session.table("analytics.documents")
    parameters = dict(
        session.indexes.load("documents_embedding").metadata["snapshots"][0][
            "parameters"
        ]
    )

    class _Builder:
        info = relify.builders.BuilderInfo(
            "fixture",
            "Fixture",
            "relify-tests",
        )
        capabilities = relify.builders.BuilderCapabilities(
            frozenset(
                {
                    relify.builders.BuildProfile(
                        "ivf",
                        "iceberg",
                        "iceberg",
                    )
                }
            )
        )

        def build(
            self,
            request: relify.builders.BuildRequest,
            context: relify.builders.BuildContext,
        ) -> relify.builders.BuildOutput:
            assert request.source["snapshot-id"] == 101
            assert context.iceberg_catalog is catalog
            return relify.builders.BuildOutput(
                parameters=parameters,
                index_relations={
                    "ivf_centroids": _relation(
                        catalog,
                        ("relify", "documents_embedding_centroids"),
                    ),
                    "ivf_postings": _relation(
                        catalog,
                        ("relify", "documents_embedding_postings"),
                    ),
                },
            )

    documents.create_index(
        "other_embedding",
        column="embedding",
        key=["document_id"],
        config=relify.IVF(nlist=2),
        builder=_Builder(),
        wait_timeout=timedelta(seconds=5),
    )

    status = documents.index_status("other_embedding")
    assert status.state == "ready"
    assert status.builder == "fixture"


def test_starrocks_requires_one_named_iceberg_catalog(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="catalog_name"):
        relify.experimental.starrocks.connect(
            _Connection(),
            index_catalog=f"sqlite://{tmp_path / 'catalog.sqlite'}",
            iceberg_catalog=object(),
        )

    session, _, _ = _session(tmp_path)
    with pytest.raises(ValueError, match="different Iceberg catalog"):
        session.table(relify.TableIdentifier("other", ("analytics",), "documents"))


def test_starrocks_rejects_values_outside_finite_float_range(
    tmp_path: Path,
) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")

    with pytest.raises(ValueError, match="finite float"):
        session.to_sql(
            documents.search([1.0e300, 0.0], column="embedding").bypass_vector_index()
        )


def test_starrocks_exact_search_accepts_a_nullable_vector_declaration(
    tmp_path: Path,
) -> None:
    session, _, catalog = _session(tmp_path)
    source, _, _ = _schemas()
    nullable_fields = [
        (
            _field(
                field.field_id,
                field.name,
                field.field_type,
                required=False,
            )
            if field.name == "embedding"
            else field
        )
        for field in source.fields
    ]
    catalog.tables[("analytics", "documents")] = _IcebergTable(
        301,
        Schema(*nullable_fields, schema_id=4),
    )
    documents = session.table("analytics.documents")

    sql = session.to_sql(
        documents.search([0.0, 0.0], column="embedding").bypass_vector_index()
    )

    assert "embedding" in sql

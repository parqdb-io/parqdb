from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import relify
from relify.experimental.spark.iceberg import load_table_state
from relify.experimental.spark.schema import canonical_schema


@dataclass
class _Snapshot:
    snapshot_id: int
    summary: dict[str, str] | None = None


@dataclass
class _Metadata:
    table_uuid: uuid.UUID


class _IcebergTable:
    def __init__(self, table_uuid: uuid.UUID, snapshot_id: int) -> None:
        self.metadata = _Metadata(table_uuid)
        self._snapshot = _Snapshot(snapshot_id, {"total-records": "3"})

    def refresh(self) -> None:
        pass

    def current_snapshot(self) -> _Snapshot:
        return self._snapshot

    def snapshot_by_id(self, snapshot_id: int) -> _Snapshot | None:
        return self._snapshot if snapshot_id == self._snapshot.snapshot_id else None


class _IcebergCatalog:
    name = "lakehouse"

    def __init__(self) -> None:
        self.tables: dict[tuple[str, ...], _IcebergTable] = {
            ("analytics", "documents"): _IcebergTable(uuid.uuid4(), 101)
        }
        self.namespaces = {("analytics",)}

    def load_table(self, identifier: tuple[str, ...]) -> _IcebergTable:
        return self.tables[identifier]

    def namespace_exists(self, namespace: tuple[str, ...]) -> bool:
        return namespace in self.namespaces

    def create_namespace(self, namespace: tuple[str, ...]) -> None:
        self.namespaces.add(namespace)

    def purge_table(self, identifier: tuple[str, ...]) -> None:
        self.tables.pop(identifier)


class _SparkContext:
    defaultParallelism = 4

    def setJobGroup(
        self,
        group_id: str,
        description: str,
        interrupt_on_cancel: bool,
    ) -> None:
        pass

    def setLocalProperty(self, key: str, value: str | None) -> None:
        pass


def _index_parameters(tmp_path: Path) -> dict[str, str]:
    return {
        "dimension": "2",
        "nlist": "2",
        "ntotal": "3",
        "posting_encoding": "source",
        "shared_ivf_fingerprint": "33333333-3333-3333-3333-333333333333",
        "shared_ivf_uuid": "44444444-4444-4444-4444-444444444444",
        "shared_ivf_metadata_location": (
            tmp_path / "shared" / "v1.metadata.json"
        ).as_uri(),
    }


class _SparkSession:
    def __init__(self) -> None:
        self.sparkContext = _SparkContext()
        self.table_names: list[str] = []

    def table(self, identifier: str) -> object:
        self.table_names.append(identifier)
        return object()


def _session(tmp_path: Path) -> tuple[Any, _SparkSession, _IcebergCatalog]:
    spark = _SparkSession()
    iceberg = _IcebergCatalog()
    session = relify.experimental.spark.connect(
        spark,
        index_catalog=f"sqlite://{tmp_path / 'catalog.sqlite'}",
        iceberg_catalog=iceberg,
    )
    return session, spark, iceberg


def test_spark_session_resolves_the_same_table_in_both_catalogs(tmp_path: Path) -> None:
    session, spark, _ = _session(tmp_path)

    documents = session.table("analytics.documents")

    assert isinstance(documents, relify.experimental.spark.Table)
    assert documents.identifier == relify.TableIdentifier(
        "lakehouse",
        ("analytics",),
        "documents",
    )
    assert spark.table_names == ["`lakehouse`.`analytics`.`documents`"]
    query = documents.search([0.0, 1.0], column="embedding").nprobes(4).limit(10)
    assert query.source == documents.identifier
    assert query.probe_count == 4
    assert session.default_builder is None


def test_spark_create_index_forwards_write_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")
    submitted: dict[str, Any] = {}

    def create(source_identifier: relify.TableIdentifier, **kwargs: Any) -> None:
        submitted["source_identifier"] = source_identifier
        submitted.update(kwargs)

    monkeypatch.setattr(session._builds, "create", create)
    options = relify.WriteOptions(partitions=17)
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["document_id"],
        config=relify.IVF(nlist=2),
        writer_options=options,
    )

    assert submitted["writer_options"] == options
    assert submitted["builder"] is None


def test_spark_session_collects_portable_arrow_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")
    query = documents.search([0.0, 1.0], column="embedding")
    expected = pa.table({"document_id": [1, 2]})

    class _DataFrame:
        def toArrow(self) -> pa.Table:
            return expected

    monkeypatch.setattr(
        "relify.experimental.spark.session.plan_query", lambda *_: _DataFrame()
    )

    assert session.collect(query) == expected


def test_spark_session_explains_through_the_public_dataframe_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")
    query = documents.search([0.0, 1.0], column="embedding")

    class _DataFrame:
        def explain(self, *, extended: bool) -> None:
            print(f"Spark plan extended={extended}")

    monkeypatch.setattr(
        "relify.experimental.spark.session.plan_query", lambda *_: _DataFrame()
    )

    assert session.explain(query, verbose=True) == "Spark plan extended=True"


def test_spark_catalog_facade_uses_shared_native_repository(tmp_path: Path) -> None:
    session, _, _ = _session(tmp_path)
    documents = session.table("analytics.documents")
    source = session._load_source_state(documents.identifier)
    relation = {
        "profile": "iceberg",
        "catalog": "lakehouse",
        "namespace": ["relify"],
        "name": "documents_embedding_centroids",
        "table-uuid": str(uuid.uuid4()),
        "snapshot-id": 201,
    }
    session._native.publish_initial(
        index_name="documents_embedding",
        source_json=source.relation_json(),
        vector_field="embedding",
        source_key_fields=["document_id"],
        builder="fixture",
        metric="l2_squared",
        parameters=_index_parameters(tmp_path),
        index_relations={
            "ivf_centroids": json.dumps(relation),
            "ivf_postings": json.dumps(
                {
                    **relation,
                    "name": "documents_embedding_postings",
                    "table-uuid": str(uuid.uuid4()),
                    "snapshot-id": 202,
                }
            ),
        },
    )

    assert session.indexes.list() == ["documents_embedding"]
    entry = session.indexes.load("documents_embedding")
    assert entry.metadata["snapshots"][0]["summary"]["builder"] == "fixture"
    assert documents.list_indexes()[0].name == "documents_embedding"
    assert documents.index_status("documents_embedding").state == "ready"

    documents.drop_index("documents_embedding")
    assert session.indexes.list() == []
    with pytest.raises(relify.IndexNotFoundError, match="index not found"):
        documents.drop_index("documents_embedding")


def test_spark_relation_dispatch_reads_parquet_uri() -> None:
    from relify.experimental.spark import planner

    expected = object()

    class _Read:
        def parquet(self, uri: str) -> object:
            assert uri == "s3://bucket/index/postings"
            return expected

    spark = type("_Spark", (), {"read": _Read()})()
    session = type("_Session", (), {"spark": spark})()

    assert (
        planner._read_relation(
            session,
            {
                "profile": "parquet",
                "uri": "s3://bucket/index/postings",
            },
        )
        is expected
    )


def test_spark_can_mark_inferred_parquet_nullability_as_unknown() -> None:
    spark_types = pytest.importorskip("pyspark.sql.types")
    inferred = spark_types.StructType(
        [
            spark_types.StructField("key_1", spark_types.LongType(), nullable=True),
            spark_types.StructField(
                "vector",
                spark_types.ArrayType(
                    spark_types.FloatType(),
                    containsNull=True,
                ),
                nullable=True,
            ),
            spark_types.StructField("cid", spark_types.IntegerType(), nullable=True),
        ]
    )

    schema = canonical_schema(
        inferred,
        nullability_known=False,
    )

    assert schema.names == ("key_1", "vector", "cid")
    assert not schema.nullability_known
    assert not any(field.required for field in schema.fields)
    assert not schema.field("vector").field_type.element_required


def test_spark_session_registers_parquet_without_iceberg(
    tmp_path: Path,
) -> None:
    class _Read:
        def __init__(self) -> None:
            self.uris: list[str] = []

        def parquet(self, uri: str) -> object:
            self.uris.append(uri)
            return object()

    spark = _SparkSession()
    spark.read = _Read()  # type: ignore[attr-defined]
    session = relify.experimental.spark.connect(
        spark,
        index_catalog=f"sqlite://{tmp_path / 'catalog.sqlite'}",
    )

    documents = session.register_parquet(
        "documents",
        "file:///data/documents/*.parquet",
    )

    assert documents.identifier == relify.TableIdentifier(
        "relify",
        ("parquet",),
        "documents",
    )
    assert session._load_source_state(documents.identifier).relation_json() == (
        '{"profile":"parquet","uri":"file:///data/documents/*.parquet"}'
    )
    assert spark.read.uris == ["file:///data/documents/*.parquet"]


def test_spark_connect_is_rejected_explicitly(tmp_path: Path) -> None:
    class _Connect:
        @property
        def sparkContext(self) -> None:
            raise RuntimeError("Spark Connect has no SparkContext")

    with pytest.raises(NotImplementedError, match="Spark Classic"):
        relify.experimental.spark.connect(
            _Connect(),
            index_catalog=f"sqlite://{tmp_path / 'catalog.sqlite'}",
            iceberg_catalog=_IcebergCatalog(),
        )


@pytest.mark.requires("spark")
def test_native_pyspark_plan_executes_index_only_and_source_join_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyspark")
    spark_sql = pytest.importorskip("pyspark.sql")
    spark_types = pytest.importorskip("pyspark.sql.types")
    planner = pytest.importorskip("relify.experimental.spark.planner")
    spark = (
        spark_sql.SparkSession.builder.master("local[2]")
        .appName("relify-spark-plan-test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    iceberg = _IcebergCatalog()
    identifiers = {
        "documents": relify.TableIdentifier("lakehouse", ("analytics",), "documents"),
        "centroids": relify.TableIdentifier(
            "lakehouse",
            ("relify",),
            "documents_embedding_centroids",
        ),
        "postings": relify.TableIdentifier(
            "lakehouse",
            ("relify",),
            "documents_embedding_postings",
        ),
    }
    iceberg.tables[("relify", "documents_embedding_centroids")] = _IcebergTable(
        uuid.uuid4(),
        201,
    )
    iceberg.tables[("relify", "documents_embedding_postings")] = _IcebergTable(
        uuid.uuid4(),
        202,
    )
    frames = {
        "documents": spark.createDataFrame(
            [
                (1, [0.0, 0.0], 1, "a"),
                (2, [1.0, 0.0], 0, "b"),
                (3, [10.0, 0.0], 1, "c"),
            ],
            schema=spark_types.StructType(
                [
                    spark_types.StructField(
                        "document_id",
                        spark_types.LongType(),
                        nullable=False,
                    ),
                    spark_types.StructField(
                        "embedding",
                        spark_types.ArrayType(
                            spark_types.FloatType(),
                            containsNull=False,
                        ),
                        nullable=False,
                    ),
                    spark_types.StructField(
                        "tenant_id",
                        spark_types.IntegerType(),
                        nullable=False,
                    ),
                    spark_types.StructField(
                        "title",
                        spark_types.StringType(),
                        nullable=False,
                    ),
                ]
            ),
        ),
        "documents_embedding_centroids": spark.createDataFrame(
            [(0, [0.5, 0.0]), (1, [10.0, 0.0])],
            schema=spark_types.StructType(
                [
                    spark_types.StructField(
                        "cid",
                        spark_types.IntegerType(),
                        nullable=False,
                    ),
                    spark_types.StructField(
                        "centroid",
                        spark_types.ArrayType(
                            spark_types.FloatType(),
                            containsNull=False,
                        ),
                        nullable=False,
                    ),
                ]
            ),
        ),
        "documents_embedding_postings": spark.createDataFrame(
            [
                (0, 1),
                (0, 2),
                (1, 3),
            ],
            schema=spark_types.StructType(
                [
                    spark_types.StructField(
                        "cid",
                        spark_types.IntegerType(),
                        nullable=False,
                    ),
                    spark_types.StructField(
                        "key_1",
                        spark_types.LongType(),
                        nullable=False,
                    ),
                ]
            ),
        ),
    }
    try:
        session = relify.experimental.spark.connect(
            spark,
            index_catalog=f"sqlite://{tmp_path / 'catalog.sqlite'}",
            iceberg_catalog=iceberg,
        )
        source = load_table_state(iceberg, identifiers["documents"])
        centroids = load_table_state(iceberg, identifiers["centroids"])
        postings = load_table_state(iceberg, identifiers["postings"])
        session._native.publish_initial(
            index_name="documents_embedding",
            source_json=source.relation_json(),
            vector_field="embedding",
            source_key_fields=["document_id"],
            builder="fixture",
            metric="l2_squared",
            parameters=_index_parameters(tmp_path),
            index_relations={
                "ivf_centroids": centroids.relation_json(),
                "ivf_postings": postings.relation_json(),
            },
        )
        monkeypatch.setattr(
            planner,
            "_read_snapshot",
            lambda _spark, identifier, _snapshot: frames[identifier.name],
        )
        documents = relify.experimental.spark.Table(session, identifiers["documents"])

        index_only = (
            documents.search([0.0, 0.0], column="embedding")
            .nprobes(1)
            .limit(10)
            .select(["document_id"])
        )
        index_only_result = session.to_dataframe(index_only)
        assert not index_only_result.schema["document_id"].nullable
        assert not index_only_result.schema["_distance"].nullable
        assert index_only_result.collect() == [
            spark_sql.Row(document_id=1, _distance=0.0),
            spark_sql.Row(document_id=2, _distance=1.0),
        ]

        source_join = (
            documents.search([0.0, 0.0], column="embedding")
            .where("tenant_id = 1")
            .nprobes(2)
            .limit(10)
            .select(["document_id", "title"])
        )
        assert session.to_dataframe(source_join).collect() == [
            spark_sql.Row(document_id=1, title="a", _distance=0.0),
            spark_sql.Row(document_id=3, title="c", _distance=100.0),
        ]

        overflowing = spark.createDataFrame(
            [([3.0e38],)],
            spark_types.StructType(
                [
                    spark_types.StructField(
                        "vector",
                        spark_types.ArrayType(
                            spark_types.FloatType(),
                            containsNull=False,
                        ),
                        nullable=False,
                    )
                ]
            ),
        ).select(
            planner._squared_l2(
                planner._functions().col("vector"),
                [-3.0e38],
            ).alias("_distance")
        )
        with pytest.raises(Exception, match="squared-L2 distance is non-finite"):
            overflowing.collect()

    finally:
        spark.stop()


@pytest.mark.requires("spark")
def test_spark_queries_parquet_index_built_by_local_backend(tmp_path: Path) -> None:
    spark_sql = pytest.importorskip("pyspark.sql")
    source_path = tmp_path / "documents.parquet"
    schema = pa.schema(
        [
            pa.field("document_id", pa.int64(), nullable=False),
            pa.field(
                "embedding",
                pa.list_(pa.field("element", pa.float32(), nullable=False)),
                nullable=False,
            ),
        ]
    )
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([1, 2, 3], type=pa.int64()),
            pa.array(
                [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]],
                type=schema.field("embedding").type,
            ),
        ],
        schema=schema,
    )
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_batches([batch], schema=schema), source_path)
    root = tmp_path / "local"
    local = relify.connect(root)
    local.register_parquet("documents", source_path)
    documents = local.table("documents")
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["document_id"],
        config=relify.IVF(nlist=2),
    )
    documents.wait_for_index("documents_embedding")

    spark = (
        spark_sql.SparkSession.builder.master("local[2]")
        .appName("relify-cross-backend-parquet-query")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        session = relify.experimental.spark.connect(
            spark,
            index_catalog=f"sqlite://{root / 'catalog.sqlite'}",
            metadata_root=root.as_uri(),
        )
        spark_documents = session.register_parquet(
            "documents",
            source_path.as_uri(),
        )
        query = (
            spark_documents.search([0.0, 0.0], column="embedding")
            .nprobes(2)
            .limit(10)
            .select(["document_id"])
        )
        assert session.to_dataframe(query).collect() == [
            spark_sql.Row(document_id=1, _distance=0.0),
            spark_sql.Row(document_id=2, _distance=1.0),
            spark_sql.Row(document_id=3, _distance=100.0),
        ]
    finally:
        spark.stop()

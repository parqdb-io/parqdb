from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from support.config import SparkConfig
from support.indexes import write_shared_ivf_metadata

pytestmark = pytest.mark.requires("spark")


class _HadoopPyIcebergCatalog:
    """Test adapter sharing one local warehouse with Spark HadoopCatalog."""

    name = "lakehouse"

    def __init__(self, warehouse: Path) -> None:
        self._warehouse = warehouse

    def load_table(self, identifier: tuple[str, ...]) -> Any:
        from pyiceberg.table import StaticTable

        metadata = self._table_path(identifier) / "metadata"
        version = int((metadata / "version-hint.text").read_text().strip())
        return StaticTable.from_metadata(
            (metadata / f"v{version}.metadata.json").as_uri()
        )

    def namespace_exists(self, namespace: tuple[str, ...]) -> bool:
        return self._warehouse.joinpath(*namespace).is_dir()

    def list_namespaces(self) -> list[tuple[str, ...]]:
        return []

    def create_namespace(self, namespace: tuple[str, ...]) -> None:
        self._warehouse.joinpath(*namespace).mkdir(
            parents=True,
            exist_ok=False,
        )

    def create_table(
        self,
        identifier: tuple[str, ...],
        *,
        schema: Any,
        properties: dict[str, str],
    ) -> Any:
        from pyiceberg.io import load_file_io
        from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
        from pyiceberg.serializers import ToOutputFile
        from pyiceberg.table.metadata import new_table_metadata
        from pyiceberg.table.sorting import UNSORTED_SORT_ORDER

        table = self._table_path(identifier)
        metadata_directory = table / "metadata"
        metadata_directory.mkdir(parents=True, exist_ok=False)
        metadata = new_table_metadata(
            schema,
            UNPARTITIONED_PARTITION_SPEC,
            UNSORTED_SORT_ORDER,
            table.as_uri(),
            dict(properties),
        )
        metadata_location = (metadata_directory / "v1.metadata.json").as_uri()
        file_io = load_file_io({}, metadata_location)
        ToOutputFile.table_metadata(
            metadata,
            file_io.new_output(metadata_location),
        )
        (metadata_directory / "version-hint.text").write_text("1")
        return self.load_table(identifier)

    def purge_table(self, identifier: tuple[str, ...]) -> None:
        shutil.rmtree(self._table_path(identifier))

    def _table_path(self, identifier: tuple[str, ...]) -> Path:
        return self._warehouse.joinpath(*identifier)


def test_spark_queries_published_index_and_repository_reopens_with_real_iceberg(
    tmp_path: Path,
    spark: SparkConfig,
) -> None:
    relify = pytest.importorskip("relify")
    pyspark = pytest.importorskip("pyspark")
    spark_sql = pytest.importorskip("pyspark.sql")
    spark_types = pytest.importorskip("pyspark.sql.types")

    spark_line = ".".join(pyspark.__version__.split(".")[:2])
    if spark_line not in {"4.0", "4.1"}:
        pytest.skip(f"Iceberg runtime coordinate is not defined for Spark {spark_line}")
    runtime = (
        "org.apache.iceberg:"
        f"iceberg-spark-runtime-{spark_line}_2.13:{spark.iceberg_version}"
    )
    warehouse = tmp_path / "warehouse"
    spark = (
        spark_sql.SparkSession.builder.master("local[2]")
        .appName("relify-spark-iceberg-integration")
        .config("spark.ui.enabled", "false")
        .config("spark.jars.packages", runtime)
        .config(
            "spark.sql.catalog.lakehouse",
            "org.apache.iceberg.spark.SparkCatalog",
        )
        .config("spark.sql.catalog.lakehouse.type", "hadoop")
        .config("spark.sql.catalog.lakehouse.warehouse", warehouse.as_uri())
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        spark.sql("CREATE NAMESPACE lakehouse.analytics")
        spark.sql("CREATE NAMESPACE lakehouse.relify")
        from pyiceberg.schema import Schema
        from pyiceberg.types import (
            FloatType,
            IntegerType,
            ListType,
            LongType,
            NestedField,
            StringType,
        )

        catalog = _HadoopPyIcebergCatalog(warehouse)
        catalog.create_table(
            ("analytics", "documents"),
            schema=Schema(
                NestedField(1, "document_id", LongType(), required=True),
                NestedField(
                    2,
                    "embedding",
                    ListType(3, FloatType(), element_required=True),
                    required=True,
                ),
                NestedField(4, "tenant_id", IntegerType(), required=True),
                NestedField(5, "title", StringType(), required=True),
            ),
            properties={},
        )
        catalog.create_table(
            ("relify", "documents_embedding_centroids"),
            schema=Schema(
                NestedField(1, "cid", IntegerType(), required=True),
                NestedField(
                    2,
                    "centroid",
                    ListType(3, FloatType(), element_required=True),
                    required=True,
                ),
            ),
            properties={},
        )
        catalog.create_table(
            ("relify", "documents_embedding_postings"),
            schema=Schema(
                NestedField(1, "cid", IntegerType(), required=True),
                NestedField(2, "key_1", LongType(), required=True),
            ),
            properties={},
        )
        source_schema = spark_types.StructType(
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
        )
        spark.createDataFrame(
            [
                (1, [0.0, 0.0], 42, "a"),
                (2, [1.0, 0.0], 42, "b"),
                (3, [10.0, 0.0], 7, "c"),
            ],
            source_schema,
        ).writeTo("lakehouse.analytics.documents").append()

        centroids_schema = spark_types.StructType(
            [
                spark_types.StructField("cid", spark_types.IntegerType(), False),
                spark_types.StructField(
                    "centroid",
                    spark_types.ArrayType(spark_types.FloatType(), False),
                    False,
                ),
            ]
        )
        spark.createDataFrame(
            [(0, [0.5, 0.0]), (1, [10.0, 0.0])],
            centroids_schema,
        ).writeTo("lakehouse.relify.documents_embedding_centroids").append()

        postings_schema = spark_types.StructType(
            [
                spark_types.StructField("cid", spark_types.IntegerType(), False),
                spark_types.StructField("key_1", spark_types.LongType(), False),
            ]
        )
        spark.createDataFrame(
            [(0, 1), (0, 2), (1, 3)],
            postings_schema,
        ).writeTo("lakehouse.relify.documents_embedding_postings").append()

        catalog_uri = f"sqlite://{tmp_path / 'relify.sqlite'}"
        metadata_root = tmp_path / "relify-metadata"
        session = relify.experimental.spark.connect(
            spark,
            index_catalog=catalog_uri,
            iceberg_catalog=catalog,
            metadata_root=metadata_root.as_uri(),
        )
        source = _relation(
            catalog.load_table(("analytics", "documents")),
            ("analytics", "documents"),
        )
        centroids = _relation(
            catalog.load_table(("relify", "documents_embedding_centroids")),
            ("relify", "documents_embedding_centroids"),
        )
        shared = write_shared_ivf_metadata(
            metadata_root,
            source=source,
            centroids=centroids,
        )
        session._native.publish_initial(
            index_name="documents_embedding",
            source_json=json.dumps(source, separators=(",", ":")),
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
                "ivf_centroids": json.dumps(centroids, separators=(",", ":")),
                "ivf_postings": json.dumps(
                    _relation(
                        catalog.load_table(("relify", "documents_embedding_postings")),
                        ("relify", "documents_embedding_postings"),
                    ),
                    separators=(",", ":"),
                ),
            },
        )

        postings_schema = catalog.load_table(
            ("relify", "documents_embedding_postings")
        ).schema()
        assert postings_schema.find_field("cid").required
        assert postings_schema.find_field("key_1").required
        assert "vector" not in {field.name for field in postings_schema.fields}

        reopened = relify.experimental.spark.connect(
            spark,
            index_catalog=catalog_uri,
            iceberg_catalog=catalog,
            metadata_root=metadata_root.as_uri(),
        )
        reopened_documents = reopened.table("analytics.documents")
        query = (
            reopened_documents.search([0.0, 0.0], column="embedding")
            .where("tenant_id = 42")
            .nprobes(2)
            .limit(10)
            .select(["document_id", "title"])
        )
        result = reopened.to_dataframe(query)
        assert not result.schema["document_id"].nullable
        assert not result.schema["title"].nullable
        assert not result.schema["_distance"].nullable
        assert result.collect() == [
            spark_sql.Row(document_id=1, title="a", _distance=0.0),
            spark_sql.Row(document_id=2, title="b", _distance=1.0),
        ]

        datafusion = relify.connect(
            catalog=catalog_uri,
            index_root=metadata_root.as_uri(),
            iceberg=catalog,
        )
        datafusion_documents = datafusion.table("lakehouse.analytics.documents")
        datafusion_result = datafusion.to_arrow(
            datafusion_documents.search(
                [0.0, 0.0],
                column="embedding",
            )
            .where("tenant_id = 42")
            .nprobes(2)
            .limit(10)
            .select(["document_id", "title"])
        )
        assert datafusion_result.to_pydict() == {
            "document_id": [1, 2],
            "title": ["a", "b"],
            "_distance": [0.0, 1.0],
        }
    finally:
        spark.stop()


def _relation(table: Any, identifier: tuple[str, ...]) -> dict[str, object]:
    snapshot = table.current_snapshot()
    assert snapshot is not None
    return {
        "profile": "iceberg",
        "catalog": "lakehouse",
        "namespace": list(identifier[:-1]),
        "name": identifier[-1],
        "table-uuid": str(table.metadata.table_uuid).lower(),
        "snapshot-id": int(snapshot.snapshot_id),
    }

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from support.config import SparkConfig

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


def test_spark_build_query_and_repository_reopen_with_real_iceberg(
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
        ).writeTo("lakehouse.analytics.documents").using("iceberg").create()

        catalog = _HadoopPyIcebergCatalog(warehouse)
        catalog_uri = f"sqlite://{tmp_path / 'relify.sqlite'}"
        session = relify.experimental.spark.connect(
            spark,
            index_catalog=catalog_uri,
            iceberg_catalog=catalog,
        )
        documents = session.table("analytics.documents")
        documents.create_index(
            "documents_embedding",
            column="embedding",
            key=["document_id"],
            config=relify.IVF(nlist=2),
        )
        documents.wait_for_index("documents_embedding")

        postings_schema = catalog.load_table(
            ("relify", "documents_embedding_postings")
        ).schema()
        assert postings_schema.find_field("cid").required
        assert postings_schema.find_field("key_1").required
        vector = postings_schema.find_field("vector")
        assert vector.required
        assert vector.field_type.element_required

        reopened = relify.experimental.spark.connect(
            spark,
            index_catalog=catalog_uri,
            iceberg_catalog=catalog,
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
        assert result.schema["document_id"].nullable
        assert result.schema["title"].nullable
        assert not result.schema["_distance"].nullable
        assert result.collect() == [
            spark_sql.Row(document_id=1, title="a", _distance=0.0),
            spark_sql.Row(document_id=2, title="b", _distance=1.0),
        ]

        datafusion = relify.connect(
            catalog=catalog_uri,
            index_root=(tmp_path / "relify-metadata").as_uri(),
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

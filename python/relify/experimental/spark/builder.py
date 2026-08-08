from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from ...builders.v1 import BuildContext, BuildOutput, BuildRequest
from ...config import IVF, WriteOptions
from ...identifier import TableIdentifier
from .iceberg import (
    ensure_namespace,
    load_table_state,
    read_snapshot,
    spark_identifier,
    validate_relation,
)

_INDEX_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ASSIGNMENT_BYTES = 64 * 1024 * 1024
_COARSE_MIN_POINTS_PER_CENTROID = 39
_COARSE_MAX_POINTS_PER_CENTROID = 256
_DEFAULT_SEED = 42
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AssignmentMetrics:
    rows: Any
    batches: Any
    conversion_ns: Any
    distance_ns: Any
    output_ns: Any

    @classmethod
    def create(cls, spark_context: Any) -> _AssignmentMetrics:
        return cls(
            rows=spark_context.accumulator(0),
            batches=spark_context.accumulator(0),
            conversion_ns=spark_context.accumulator(0),
            distance_ns=spark_context.accumulator(0),
            output_ns=spark_context.accumulator(0),
        )

    def log(self, *, source_rows: int, write_seconds: float) -> None:
        evaluated_rows = int(self.rows.value)
        worker_seconds = (
            int(self.conversion_ns.value)
            + int(self.distance_ns.value)
            + int(self.output_ns.value)
        ) / 1_000_000_000
        worker_ns_per_row = (
            worker_seconds * 1_000_000_000 / evaluated_rows
            if evaluated_rows > 0
            else 0.0
        )
        evaluations_per_source_row = (
            evaluated_rows / source_rows if source_rows > 0 else 0.0
        )
        _LOGGER.info(
            "Spark IVF postings completed: source_rows=%d "
            "assignment_rows_evaluated=%d "
            "assignment_evaluations_per_source_row=%.2f batches=%d "
            "assignment_worker_seconds=%.3f conversion_worker_seconds=%.3f "
            "distance_worker_seconds=%.3f output_worker_seconds=%.3f "
            "assignment_worker_ns_per_row=%.1f "
            "assignment_shuffle_write_wall_seconds=%.3f",
            source_rows,
            evaluated_rows,
            evaluations_per_source_row,
            int(self.batches.value),
            worker_seconds,
            int(self.conversion_ns.value) / 1_000_000_000,
            int(self.distance_ns.value) / 1_000_000_000,
            int(self.output_ns.value) / 1_000_000_000,
            worker_ns_per_row,
            write_seconds,
        )


def build_initial(
    spark: Any,
    request: BuildRequest,
    context: BuildContext,
) -> BuildOutput:
    if context.iceberg_catalog is None or not isinstance(
        context.catalog_name,
        str,
    ):
        raise ValueError("Spark index construction requires an Iceberg catalog")
    if request.profile != "iceberg":
        raise NotImplementedError(
            "the first Spark builder requires an Iceberg source table"
        )
    source = validate_relation(
        context.iceberg_catalog,
        dict(request.source),
    )
    index = request.index
    column = request.column
    key = list(request.key)
    config = request.config
    writer_options = request.writer_options
    _validate_request(index, column, key, config, writer_options)
    posting_encoding = config.encoding
    if posting_encoding not in {"source", "flat"}:
        raise NotImplementedError(
            "the Spark builder does not support quantized IVF postings"
        )
    store_vectors = posting_encoding == "flat"
    modules = _spark_modules()
    source_df = read_snapshot(
        spark,
        source.identifier,
        source.snapshot_id,
    )
    created: list[tuple[str, ...]] = []
    try:
        key_types = _validate_iceberg_schema(
            context.iceberg_catalog,
            source.identifier,
            source.snapshot_id,
            column=column,
            key=key,
        )
        _validate_source_schema(
            source_df,
            column=column,
            key=key,
        )
        ntotal = _snapshot_row_count(
            context.iceberg_catalog,
            source.identifier,
            source.snapshot_id,
        )
        if ntotal is None:
            _LOGGER.warning(
                "Iceberg snapshot %s has no total-records summary; "
                "falling back to a Spark count",
                source.snapshot_id,
            )
            ntotal = int(source_df.count())
        if config.nlist > ntotal:
            raise ValueError("nlist must not exceed the number of source rows")
        centroids = _train_centroids(
            source_df,
            column=column,
            nlist=config.nlist,
            ntotal=ntotal,
            modules=modules,
        )
        dimension = int(centroids.shape[1])
        centroids_df = _centroid_dataframe(
            spark,
            centroids,
            modules=modules,
        )
        centroid_schema, postings_schema = _index_schemas(
            key_types,
            store_vectors=store_vectors,
        )

        ensure_namespace(context.iceberg_catalog, context.index_namespace)
        catalog_name = context.catalog_name
        if not isinstance(catalog_name, str):
            raise ValueError("Spark index construction requires an Iceberg catalog")
        centroid_identifier = TableIdentifier(
            catalog_name,
            context.index_namespace,
            f"{index}_centroids",
        )
        postings_identifier = TableIdentifier(
            catalog_name,
            context.index_namespace,
            f"{index}_postings",
        )
        _create_iceberg_table(
            context.iceberg_catalog,
            centroids_df,
            centroid_identifier,
            centroid_schema,
            writer_options,
        )
        created.append((*centroid_identifier.namespace, centroid_identifier.name))
        spark_context = spark.sparkContext
        centroid_norms = modules["numpy"].einsum(
            "ij,ij->i",
            centroids,
            centroids,
        )
        centroid_broadcast = spark_context.broadcast((centroids, centroid_norms))
        assignment_metrics = _AssignmentMetrics.create(spark_context)
        try:
            postings = _assign_postings(
                source_df,
                column=column,
                key=key,
                centroids=centroid_broadcast,
                centroids_are_broadcast=True,
                store_vectors=store_vectors,
                modules=modules,
                metrics=assignment_metrics,
            )
            started = time.perf_counter()
            _create_iceberg_table(
                context.iceberg_catalog,
                _range_partition_postings(
                    postings,
                    key_count=len(key),
                    partitions=_posting_partitions(
                        spark_context,
                        writer_options.partitions,
                    ),
                ),
                postings_identifier,
                postings_schema,
                writer_options,
            )
            created.append((*postings_identifier.namespace, postings_identifier.name))
            assignment_metrics.log(
                source_rows=ntotal,
                write_seconds=time.perf_counter() - started,
            )
        finally:
            centroid_broadcast.destroy()

        centroid_state = load_table_state(
            context.iceberg_catalog,
            centroid_identifier,
        )
        postings_state = load_table_state(
            context.iceberg_catalog,
            postings_identifier,
        )
        return BuildOutput(
            parameters={
                "dimension": str(dimension),
                "nlist": str(config.nlist),
                "ntotal": str(ntotal),
                "store_vectors": str(store_vectors).lower(),
            },
            index_relations={
                "ivf_centroids": centroid_state.relation_dict(),
                "ivf_postings": postings_state.relation_dict(),
            },
            discard=lambda: _purge_unpublished(
                context.iceberg_catalog,
                created,
            ),
        )
    except BaseException:
        _purge_unpublished(context.iceberg_catalog, created)
        raise


def _validate_source_schema(
    source: Any,
    *,
    column: str,
    key: list[str],
) -> None:
    fields = {field.name: field for field in source.schema.fields}
    if column not in fields:
        raise ValueError(f"unknown vector column: {column}")
    if fields[column].dataType.simpleString() != "array<float>":
        raise TypeError("the vector column must have Spark type array<float>")
    missing = [field for field in key if field not in fields]
    if missing:
        raise ValueError(f"unknown key fields: {missing}")
    if "_distance" in fields:
        raise ValueError("source table must not contain reserved column _distance")


def _validate_iceberg_schema(
    catalog: Any,
    identifier: TableIdentifier,
    snapshot_id: int,
    *,
    column: str,
    key: list[str],
) -> list[Any]:
    try:
        types = import_module("pyiceberg.types")
    except ImportError as error:
        raise ImportError(
            "Spark support requires the 'spark' extra: pip install 'relify[spark]'"
        ) from error
    table = catalog.load_table((*identifier.namespace, identifier.name))
    snapshot = table.snapshot_by_id(snapshot_id)
    if snapshot is None:
        raise ValueError(
            f"Iceberg snapshot is no longer available: {identifier!r} @ {snapshot_id}"
        )
    schema = (
        table.schemas()[snapshot.schema_id]
        if snapshot.schema_id is not None
        else table.schema()
    )
    try:
        vector = schema.find_field(column)
    except ValueError as error:
        raise ValueError(f"unknown vector column: {column}") from error
    vector_type = vector.field_type
    if not isinstance(vector_type, types.ListType) or not isinstance(
        vector_type.element_type, types.FloatType
    ):
        raise TypeError("the vector column must have Iceberg type list<float>")

    supported_keys = (
        types.BooleanType,
        types.IntegerType,
        types.LongType,
        types.BinaryType,
        types.StringType,
        types.DateType,
    )
    key_types = []
    for name in key:
        try:
            field = schema.find_field(name)
        except ValueError as error:
            raise ValueError(f"unknown key field: {name}") from error
        if isinstance(field.field_type, types.FixedType):
            raise NotImplementedError(
                "Spark construction does not yet preserve Iceberg fixed key types"
            )
        if not isinstance(field.field_type, supported_keys):
            raise TypeError(
                f"unsupported Iceberg key type for {name}: {field.field_type}"
            )
        key_types.append(field.field_type)
    return key_types


def _snapshot_row_count(
    catalog: Any,
    identifier: TableIdentifier,
    snapshot_id: int,
) -> int | None:
    table = catalog.load_table((*identifier.namespace, identifier.name))
    snapshot = table.snapshot_by_id(snapshot_id)
    if snapshot is None:
        raise ValueError(
            f"Iceberg snapshot is no longer available: {identifier!r} @ {snapshot_id}"
        )
    summary = getattr(snapshot, "summary", None)
    if summary is None:
        return None
    value = summary.get("total-records")
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Iceberg snapshot has an invalid total-records summary: {value!r}"
        ) from error
    if count < 0:
        raise ValueError(
            f"Iceberg snapshot has a negative total-records summary: {count}"
        )
    return count


def _index_schemas(
    key_types: Sequence[Any],
    *,
    store_vectors: bool,
) -> tuple[Any, Any]:
    try:
        schema_module = import_module("pyiceberg.schema")
        types = import_module("pyiceberg.types")
    except ImportError as error:
        raise ImportError(
            "Spark support requires the 'spark' extra: pip install 'relify[spark]'"
        ) from error
    centroid_schema = schema_module.Schema(
        types.NestedField(1, "cid", types.IntegerType(), required=True),
        types.NestedField(
            2,
            "centroid",
            types.ListType(3, types.FloatType(), element_required=True),
            required=True,
        ),
    )
    posting_fields = [
        types.NestedField(1, "cid", types.IntegerType(), required=True),
        *(
            types.NestedField(position + 1, f"key_{position}", key_type, required=True)
            for position, key_type in enumerate(key_types, start=1)
        ),
    ]
    if store_vectors:
        vector_id = len(key_types) + 2
        posting_fields.append(
            types.NestedField(
                vector_id,
                "vector",
                types.ListType(
                    vector_id + 1,
                    types.FloatType(),
                    element_required=True,
                ),
                required=True,
            )
        )
    return centroid_schema, schema_module.Schema(*posting_fields)


def _train_centroids(
    source: Any,
    *,
    column: str,
    nlist: int,
    ntotal: int,
    modules: dict[str, Any],
) -> Any:
    training = source
    minimum = nlist * _COARSE_MIN_POINTS_PER_CENTROID
    if ntotal < minimum:
        _LOGGER.warning(
            "training %d points to %d centroids; Faiss recommends at least %d",
            ntotal,
            nlist,
            minimum,
        )
    sample_fraction = _training_sample_fraction(ntotal, nlist)
    if sample_fraction is not None:
        training = training.sample(
            withReplacement=False,
            fraction=sample_fraction,
            seed=_DEFAULT_SEED,
        )
    features = training.select(
        modules["array_to_vector"](
            modules["functions"].col(column),
        ).alias("features")
    )
    model = modules["KMeans"](
        k=nlist,
        seed=_DEFAULT_SEED,
        maxIter=20,
        tol=1.0e-4,
        featuresCol="features",
        predictionCol="_relify_prediction",
        solver="block",
    ).fit(features)
    numpy = modules["numpy"]
    centroids = numpy.asarray(model.clusterCenters(), dtype=numpy.float32)
    if centroids.ndim != 2 or not numpy.isfinite(centroids).all():
        raise RuntimeError("Spark KMeans returned invalid centroids")
    return centroids


def _training_sample_fraction(ntotal: int, nlist: int) -> float | None:
    if ntotal < nlist:
        raise ValueError("nlist must not exceed the number of source rows")
    maximum = nlist * _COARSE_MAX_POINTS_PER_CENTROID
    return maximum / ntotal if ntotal > maximum else None


def _centroid_dataframe(spark: Any, centroids: Any, *, modules: dict[str, Any]) -> Any:
    types = modules["types"]
    schema = types.StructType(
        [
            types.StructField("cid", types.IntegerType(), nullable=False),
            types.StructField(
                "centroid",
                types.ArrayType(types.FloatType(), containsNull=False),
                nullable=False,
            ),
        ]
    )
    rows = [
        (cid, [float(value) for value in centroid])
        for cid, centroid in enumerate(centroids)
    ]
    return spark.createDataFrame(rows, schema=schema)


def _assign_postings(
    source: Any,
    *,
    column: str,
    key: list[str],
    centroids: Any,
    centroids_are_broadcast: bool = False,
    store_vectors: bool,
    modules: dict[str, Any],
    metrics: _AssignmentMetrics | None = None,
) -> Any:
    functions = modules["functions"]
    selected = source.select(
        *(
            functions.col(field).alias(f"key_{position}")
            for position, field in enumerate(key, 1)
        ),
        functions.col(column).alias("vector"),
    )
    fields = list(selected.schema.fields)
    key_fields = [
        modules["types"].StructField(
            field.name,
            field.dataType,
            nullable=False,
            metadata=field.metadata,
        )
        for field in fields[:-1]
    ]
    output_fields = [
        *key_fields,
        modules["types"].StructField(
            "cid",
            modules["types"].IntegerType(),
            nullable=False,
        ),
    ]
    if store_vectors:
        output_fields.append(
            modules["types"].StructField(
                "vector",
                modules["types"].ArrayType(
                    modules["types"].FloatType(),
                    containsNull=False,
                ),
                nullable=False,
            )
        )
    output_schema = modules["types"].StructType(output_fields)
    local_centroid_payload = (
        None
        if centroids_are_broadcast
        else (
            centroids,
            modules["numpy"].einsum("ij,ij->i", centroids, centroids),
        )
    )

    def assign(batches: Iterator[Any]) -> Iterator[Any]:
        payload = centroids.value if centroids_are_broadcast else local_centroid_payload
        if payload is None:
            raise RuntimeError("centroid payload is unavailable")
        worker_centroids, worker_centroid_norms = payload
        yield from _assignment_batches(
            batches,
            centroids=worker_centroids,
            centroid_norms=worker_centroid_norms,
            key_count=len(key),
            store_vectors=store_vectors,
            metrics=metrics,
        )

    return selected.mapInArrow(assign, schema=output_schema).select(
        "cid",
        *(f"key_{position}" for position in range(1, len(key) + 1)),
        *(["vector"] if store_vectors else []),
    )


def _range_partition_postings(
    postings: Any,
    *,
    key_count: int,
    partitions: int,
) -> Any:
    if key_count <= 0:
        raise ValueError("postings require at least one key column")
    _validate_partitions(partitions)
    order = ("cid", *(f"key_{position}" for position in range(1, key_count + 1)))
    return postings.repartitionByRange(
        partitions,
        *order,
    ).sortWithinPartitions(*order)


def _posting_partitions(context: Any, partitions: int | None) -> int:
    _validate_partitions(partitions)
    if partitions is None:
        return max(1, int(context.defaultParallelism))
    return partitions


def _validate_partitions(partitions: int | None) -> None:
    if partitions is not None and (
        not isinstance(partitions, int)
        or isinstance(partitions, bool)
        or partitions <= 0
    ):
        raise ValueError("partitions must be a positive integer")


def _assignment_batches(
    batches: Iterator[Any],
    *,
    centroids: Any,
    centroid_norms: Any | None = None,
    key_count: int,
    store_vectors: bool,
    metrics: _AssignmentMetrics | None = None,
) -> Iterator[Any]:
    numpy = import_module("numpy")
    pyarrow = import_module("pyarrow")
    resolved_centroid_norms = (
        numpy.einsum("ij,ij->i", centroids, centroids)
        if centroid_norms is None
        else centroid_norms
    )
    max_rows = max(1, _ASSIGNMENT_BYTES // max(4 * len(centroids), 4))
    for batch in batches:
        started = time.perf_counter_ns()
        vectors = _arrow_vectors(batch.column(key_count), len(centroids[0]), numpy)
        converted = time.perf_counter_ns()
        assignments = numpy.empty(len(vectors), dtype=numpy.int32)
        for start in range(0, len(vectors), max_rows):
            chunk = vectors[start : start + max_rows]
            distances = chunk @ centroids.T
            distances *= numpy.float32(-2.0)
            distances += resolved_centroid_norms[None, :]
            distances += numpy.einsum("ij,ij->i", chunk, chunk)[:, None]
            assignments[start : start + len(chunk)] = numpy.argmin(
                distances,
                axis=1,
            ).astype(numpy.int32, copy=False)
        assigned = time.perf_counter_ns()
        arrays = [
            *(batch.column(position) for position in range(key_count)),
            pyarrow.array(assignments, type=pyarrow.int32()),
        ]
        names = [
            *(f"key_{position}" for position in range(1, key_count + 1)),
            "cid",
        ]
        if store_vectors:
            arrays.append(batch.column(key_count))
            names.append("vector")
        output = pyarrow.RecordBatch.from_arrays(arrays, names=names)
        completed = time.perf_counter_ns()
        if metrics is not None:
            metrics.rows.add(len(vectors))
            metrics.batches.add(1)
            metrics.conversion_ns.add(converted - started)
            metrics.distance_ns.add(assigned - converted)
            metrics.output_ns.add(completed - assigned)
        yield output


def _arrow_vectors(array: Any, dimension: int, numpy: Any) -> Any:
    values = array.values.to_numpy(zero_copy_only=False).astype(
        numpy.float32, copy=False
    )
    offsets = array.offsets.to_numpy(zero_copy_only=False)
    lengths = offsets[1:] - offsets[:-1]
    if len(lengths) and not numpy.all(lengths == dimension):
        raise ValueError("source vectors changed dimension during index construction")
    start = int(offsets[0])
    stop = int(offsets[-1])
    vectors = values[start:stop].reshape(len(array), dimension)
    if not numpy.isfinite(vectors).all():
        raise ValueError("source vectors must be finite")
    return vectors


def _create_iceberg_table(
    catalog: Any,
    dataframe: Any,
    identifier: TableIdentifier,
    schema: Any,
    writer_options: WriteOptions,
) -> None:
    catalog_identifier = (*identifier.namespace, identifier.name)
    catalog.create_table(
        catalog_identifier,
        schema=schema,
        properties=_spark_write_properties(writer_options),
    )
    try:
        dataframe.writeTo(spark_identifier(identifier)).append()
    except BaseException:
        _purge_unpublished(catalog, [catalog_identifier])
        raise


def _spark_write_properties(options: WriteOptions) -> dict[str, str]:
    codec = options.compression
    level = None
    if codec.endswith(")") and "(" in codec:
        codec, level = codec[:-1].split("(", 1)
    if codec == "lz4_raw":
        codec = "lz4"
    properties = {
        "format-version": "2",
        "write.format.default": "parquet",
        "write.distribution-mode": "range",
        "write.parquet.compression-codec": codec,
        "write.target-file-size-bytes": str(options.target_file_size),
    }
    if level is not None:
        properties["write.parquet.compression-level"] = level
    return properties


def _purge_unpublished(catalog: Any, identifiers: Sequence[tuple[str, ...]]) -> None:
    for identifier in reversed(identifiers):
        try:
            catalog.purge_table(identifier)
        except Exception:
            pass


def _validate_request(
    index: str,
    column: str,
    key: list[str],
    config: IVF,
    writer_options: WriteOptions,
) -> None:
    if not isinstance(index, str) or _INDEX_NAME.fullmatch(index) is None:
        raise ValueError("index name must match [A-Za-z_][A-Za-z0-9_]*")
    if not isinstance(column, str) or not column:
        raise ValueError("column must be a non-empty string")
    if not key or any(not isinstance(field, str) or not field for field in key):
        raise ValueError("key must contain non-empty field names")
    if len(set(key)) != len(key):
        raise ValueError("key fields must be unique")
    if not isinstance(config, IVF):
        raise TypeError("Spark construction supports only relify.IVF")
    if not isinstance(writer_options, WriteOptions):
        raise TypeError("writer_options must be relify.WriteOptions")
    _spark_write_properties(writer_options)


def _spark_modules() -> dict[str, Any]:
    try:
        functions = import_module("pyspark.sql.functions")
        types = import_module("pyspark.sql.types")
        clustering = import_module("pyspark.ml.clustering")
        ml_functions = import_module("pyspark.ml.functions")
        numpy = import_module("numpy")
    except ImportError as error:
        raise ImportError(
            "Spark support requires the 'spark' extra: pip install 'relify[spark]'"
        ) from error
    return {
        "functions": functions,
        "types": types,
        "KMeans": clustering.KMeans,
        "array_to_vector": ml_functions.array_to_vector,
        "numpy": numpy,
    }

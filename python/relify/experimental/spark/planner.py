from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

from ...backends.v1 import (
    CanonicalSchema,
    ResolvedExactSearch,
    ResolvedSearch,
    resolve_exact_search,
    resolve_indexed_search,
    resolve_projection,
    resolve_vector_field,
    validate_ivf_schemas,
    validate_query,
    vector_fields,
)
from ...query import VectorQuery
from .iceberg import read_snapshot, validate_relation
from .schema import canonical_schema


def plan_query(session: Any, query: VectorQuery) -> Any:
    validate_query(query)
    source_state = session._load_source_state(query.source)
    source = _read_relation(session, source_state.relation_dict())
    source_schema = _relation_schema(source.schema, source_state.relation_dict())
    projection = resolve_projection(source_schema.names, query.projection)
    if query.bypass_index:
        exact = resolve_exact_search(
            query,
            source_relation=source_state.relation_dict(),
            projection=projection,
            vector_field=resolve_vector_field(
                requested=query.column,
                source_fields=source_schema.names,
                vector_fields=vector_fields(source_schema),
            ),
        )
        return _exact_plan(source, exact)

    selected = session.indexes.select(
        source_state.relation_dict(),
        index=query.index,
        column=query.column,
    )
    search = resolve_indexed_search(
        query,
        index=selected.identifier,
        metadata=selected.metadata,
        projection=projection,
    )
    if search.metric != "l2_squared" or search.posting_encoding != "source":
        raise NotImplementedError(
            "the experimental Spark backend supports only source-encoded L2 IVF indexes"
        )
    centroids = _read_relation(session, search.centroids_relation)
    postings = _read_relation(session, search.postings_relation)
    validate_ivf_schemas(
        search,
        source=source_schema,
        centroids=_relation_schema(centroids.schema, search.centroids_relation),
        postings=_relation_schema(postings.schema, search.postings_relation),
    )
    candidates = _prune_clusters(
        centroids,
        postings,
        search.query_vector,
        search.nprobe,
        search.nlist,
    )
    return _rank_candidates(
        candidates=candidates,
        source=source,
        search=search,
    )


def _exact_plan(
    source: Any,
    search: ResolvedExactSearch,
) -> Any:
    functions = _functions()
    filtered = (
        source.filter(search.predicate) if search.predicate is not None else source
    )
    distance = _squared_l2(
        functions.col(search.vector_field),
        search.query_vector,
    ).alias("_distance")
    return (
        filtered.select(
            *(functions.col(field) for field in search.projection),
            distance,
        )
        .orderBy(functions.col("_distance").asc())
        .limit(search.limit)
    )


def _prune_clusters(
    centroids: Any,
    postings: Any,
    query: Sequence[float],
    nprobe: int,
    nlist: int,
) -> Any:
    if nprobe == nlist:
        return postings
    functions = _functions()
    selected = (
        centroids.select(
            functions.col("cid"),
            _squared_l2(functions.col("centroid"), query).alias(
                "_relify_centroid_distance"
            ),
        )
        .orderBy(
            functions.col("_relify_centroid_distance").asc(),
            functions.col("cid").asc(),
        )
        .limit(nprobe)
        .select("cid")
    )
    return postings.join(selected, on="cid", how="left_semi")


def _rank_candidates(
    *,
    candidates: Any,
    source: Any,
    search: ResolvedSearch,
) -> Any:
    functions = _functions()
    if not search.needs_source:
        raise ValueError("source-encoded IVF searches must resolve the source relation")

    filtered_source = (
        source.filter(search.predicate) if search.predicate is not None else source
    )
    posting_alias = candidates.alias("_relify_postings")
    source_alias = filtered_source.alias("_relify_source")
    conditions = [
        posting_alias[f"key_{position}"] == source_alias[field]
        for position, field in enumerate(search.source_key_fields, start=1)
    ]
    condition = conditions[0]
    for next_condition in conditions[1:]:
        condition = condition & next_condition
    joined = posting_alias.join(source_alias, on=condition, how="inner")
    distance = _squared_l2(
        source_alias[search.vector_field], search.query_vector
    ).alias("_distance")
    return (
        joined.select(
            *(source_alias[field].alias(field) for field in search.projection),
            distance,
        )
        .orderBy(functions.col("_distance").asc())
        .limit(search.limit)
    )


def _read_relation(session: Any, relation: Mapping[str, Any]) -> Any:
    if relation.get("profile") == "parquet":
        return session.spark.read.parquet(str(relation["uri"]))
    if relation.get("profile") != "iceberg":
        raise ValueError(f"unsupported relation profile: {relation.get('profile')!r}")
    if session.iceberg_catalog is None:
        raise ValueError(
            "index metadata references Iceberg, but this session has no Iceberg catalog"
        )
    if relation.get("catalog") != session.catalog_name:
        raise ValueError(
            "index metadata references an Iceberg catalog not bound to this session"
        )
    state = validate_relation(session.iceberg_catalog, dict(relation))
    return _read_snapshot(session.spark, state.identifier, state.snapshot_id)


def _relation_schema(
    schema: Any,
    relation: Mapping[str, Any],
) -> CanonicalSchema:
    return canonical_schema(
        schema,
        nullability_known=relation.get("profile") != "parquet",
    )


def _read_snapshot(spark: Any, identifier: Any, snapshot_id: int) -> Any:
    return read_snapshot(spark, identifier, snapshot_id)


def _squared_l2(vector: Any, query: Sequence[float]) -> Any:
    functions = _functions()
    query_array = functions.array(
        *(functions.lit(float(value)).cast("float") for value in query)
    )
    terms = functions.zip_with(
        vector,
        query_array,
        lambda left, right: ((left - right) * (left - right)).cast("float"),
    )
    distance = functions.aggregate(
        terms,
        functions.lit(0.0).cast("float"),
        lambda total, value: (total + value).cast("float"),
    )
    invalid = (
        distance.isNull()
        | functions.isnan(distance)
        | (functions.abs(distance) == functions.lit(float("inf")))
    )
    checked = functions.when(
        invalid,
        functions.raise_error("squared-L2 distance is non-finite").cast("float"),
    ).otherwise(distance)
    # The fallback is unreachable because the null case raises above. It also
    # communicates the specification's non-null result to Spark's analyzer.
    return functions.coalesce(checked, functions.lit(0.0).cast("float"))


def _functions() -> Any:
    try:
        return import_module("pyspark.sql.functions")
    except ImportError as error:
        raise ImportError(
            "Spark support requires the 'spark' extra: pip install 'relify[spark]'"
        ) from error

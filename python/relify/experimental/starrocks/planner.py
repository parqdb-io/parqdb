from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...backends.v1 import (
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
from .iceberg import (
    ResolvedRelation,
    quote_identifier,
    relation_sql,
    resolve_reference,
)
from .schema import canonical_schema


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    """A StarRocks query and its ordered public result fields."""

    sql: str
    projection: tuple[str, ...]


def compile_query(session: Any, query: VectorQuery) -> CompiledQuery:
    validate_query(query)
    source = session._resolve_source(query.source)
    source_schema = canonical_schema(source.schema)
    projection = resolve_projection(source_schema.names, query.projection)
    if query.bypass_index:
        exact = resolve_exact_search(
            query,
            source_relation=source.state.relation_dict(),
            projection=projection,
            vector_field=resolve_vector_field(
                requested=query.column,
                source_fields=source_schema.names,
                vector_fields=vector_fields(source_schema),
            ),
        )
        return CompiledQuery(
            _exact_sql(source, exact),
            projection,
        )

    selected = session.indexes.select(
        source.state.relation_dict(),
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
            "the experimental StarRocks backend supports only source-encoded L2 IVF indexes"
        )
    centroids = resolve_reference(
        session.iceberg_catalog,
        session.catalog_name,
        dict(search.centroids_relation),
    )
    postings = resolve_reference(
        session.iceberg_catalog,
        session.catalog_name,
        dict(search.postings_relation),
    )
    validate_ivf_schemas(
        search,
        source=source_schema,
        centroids=canonical_schema(centroids.schema),
        postings=canonical_schema(postings.schema),
    )
    sql = _indexed_sql(
        source=source,
        centroids=centroids,
        postings=postings,
        search=search,
    )
    return CompiledQuery(sql, projection)


def _indexed_sql(
    *,
    source: ResolvedRelation,
    centroids: ResolvedRelation,
    postings: ResolvedRelation,
    search: ResolvedSearch,
) -> str:
    query_array = _array_literal(search.query_vector)
    ctes = []
    if search.nprobe == search.nlist:
        ctes.append(
            f"_relify_postings AS (\n    SELECT * FROM {relation_sql(postings)}\n)"
        )
    else:
        ctes.extend(
            [
                (
                    "_relify_clusters AS (\n"
                    "    SELECT `cid`\n"
                    f"    FROM {relation_sql(centroids)}\n"
                    "    ORDER BY "
                    f"l2_distance({query_array}, `centroid`) ASC, `cid` ASC\n"
                    f"    LIMIT {search.nprobe}\n"
                    ")"
                ),
                (
                    "_relify_postings AS (\n"
                    "    SELECT p.*\n"
                    f"    FROM {relation_sql(postings)} AS p\n"
                    "    INNER JOIN _relify_clusters AS c "
                    "ON p.`cid` = c.`cid`\n"
                    ")"
                ),
            ]
        )

    internal_distance = _internal_name(
        {*_schema_names(source.schema), *search.projection},
        "__relify_squared_l2",
    )
    if not search.needs_source:
        raise ValueError("source-encoded IVF searches must resolve the source relation")
    source_name = "_relify_source"
    source_body = f"    SELECT *\n    FROM {relation_sql(source)}"
    if search.predicate is not None:
        source_body += f"\n    WHERE ({search.predicate})"
    ctes.append(f"{source_name} AS (\n{source_body}\n)")
    outputs = ",\n        ".join(
        f"s.{quote_identifier(field)} AS {quote_identifier(field)}"
        for field in search.projection
    )
    joins = " AND ".join(
        f"p.{quote_identifier(f'key_{position}')} = s.{quote_identifier(source_field)}"
        for position, source_field in enumerate(
            search.source_key_fields,
            start=1,
        )
    )
    vector = f"s.{quote_identifier(search.vector_field)}"
    scored = (
        "_relify_scored AS (\n"
        "    SELECT\n"
        f"        {outputs},\n"
        f"        l2_distance({query_array}, {vector}) "
        f"AS {quote_identifier(internal_distance)}\n"
        "    FROM _relify_postings AS p\n"
        f"    INNER JOIN {source_name} AS s ON {joins}\n"
        ")"
    )
    ctes.append(scored)
    return _finish_sql(
        ctes,
        search.projection,
        internal_distance,
        search.limit,
    )


def _exact_sql(
    source: ResolvedRelation,
    search: ResolvedExactSearch,
) -> str:
    internal_distance = _internal_name(
        {*_schema_names(source.schema), *search.projection},
        "__relify_squared_l2",
    )
    outputs = ",\n        ".join(
        f"s.{quote_identifier(field)} AS {quote_identifier(field)}"
        for field in search.projection
    )
    body = (
        "_relify_scored AS (\n"
        "    SELECT\n"
        f"        {outputs},\n"
        f"        l2_distance({_array_literal(search.query_vector)}, "
        f"s.{quote_identifier(search.vector_field)}) "
        f"AS {quote_identifier(internal_distance)}\n"
        f"    FROM {relation_sql(source)} AS s"
    )
    if search.predicate is not None:
        body += f"\n    WHERE ({search.predicate})"
    body += "\n)"
    return _finish_sql(
        [body],
        search.projection,
        internal_distance,
        search.limit,
    )


def _finish_sql(
    ctes: list[str],
    projection: tuple[str, ...],
    internal_distance: str,
    limit: int,
) -> str:
    output = ",\n    ".join(quote_identifier(field) for field in projection)
    if output:
        output += ",\n    "
    distance = quote_identifier(internal_distance)
    return (
        "WITH\n"
        + ",\n".join(ctes)
        + "\nSELECT\n"
        + f"    {output}CAST({distance} AS FLOAT) AS `_distance`\n"
        + "FROM _relify_scored\n"
        + "ORDER BY `_distance` ASC\n"
        + f"LIMIT {limit}"
    )


def _schema_names(schema: Any) -> tuple[str, ...]:
    return tuple(str(field.name) for field in schema.fields)


def _array_literal(values: Sequence[float]) -> str:
    return "ARRAY<FLOAT>[" + ", ".join(_float_literal(value) for value in values) + "]"


def _float_literal(value: float) -> str:
    narrowed = struct.unpack("!f", struct.pack("!f", value))[0]
    literal = repr(narrowed)
    if "e" not in literal and "." not in literal:
        literal += ".0"
    return literal


def _internal_name(existing: set[str], preferred: str) -> str:
    value = preferred
    while value in existing:
        value += "_"
    return value

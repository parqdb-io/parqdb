# Core Concepts

Relify adds vector indexing and search to tables that are already available to
lakehouse compute engines. It separates the persisted index from the engine
that builds or queries it.

## Source Table

A source table contains the original rows, including the vector column and the
fields returned by a query. Relify does not ingest those rows into a private
database. The 0.1 implementation can bind:

- Parquet files through the local DataFusion or experimental Spark backend;
  and
- Iceberg tables through DataFusion, Spark, or StarRocks when a matching
  PyIceberg catalog is configured.

Every indexed source has one or more key columns. Keys connect index postings
back to source rows without introducing a Relify-specific row identifier.

## Open Vector Index

A Relify index consists of portable JSON metadata and ordinary relations. The
current IVF-Flat format stores:

- centroids used to choose candidate clusters; and
- postings that map clusters to source keys and, by default, exact vectors.

Parquet relations are used by the local builder. The Spark builder publishes
the same logical relations as Iceberg tables. The
[open index specification](../spec/README.md) defines their schemas and query
semantics independently of the Python implementation.

An open format makes the index inspectable and allows multiple compute engines
to consume it. Portability still requires every engine to reach the same
catalog, index relations, and source table.

## Index Catalog

The index catalog maps a logical index name and source table to an immutable
metadata document. Relify 0.1 uses SQLite. The catalog does not contain posting
rows or vectors; it coordinates publication and tells a backend which exact
index snapshot to open.

The shortcut:

```python
session = relify.connect("./relify-data")
```

creates a SQLite catalog and Parquet index root together. They can also be
configured independently, for example with a local SQLite catalog and an S3
index root.

## Snapshot

Each successful build or refresh publishes a new immutable index snapshot.
Readers resolve one snapshot before planning a query, so partially written
relations never become visible as the current index.

For Iceberg sources and indexes, metadata records exact table UUIDs and
snapshot IDs. For Parquet, the registered location is stable but the file set
has no table-format snapshot semantics; replacing files in place remains the
application's consistency responsibility.

## Builder

A builder creates the physical index and publishes its metadata:

- `relify.Local` builds Parquet indexes in the current process with native Rust
  training and assignment.
- `relify.experimental.spark.Spark` builds Iceberg indexes with Spark Classic.

Builders are selected independently from query backends. This is what allows a
Spark-built Iceberg index to be queried by DataFusion or StarRocks.

## Backend

A backend adapts the shared table, query, and index contracts to a compute
engine. It owns engine-specific planning and result collection:

| Backend | Source and index access | Build | Query |
| --- | --- | --- | --- |
| Local DataFusion | Parquet; Iceberg query with PyIceberg | Parquet | Stable |
| Spark Classic | Parquet query; Iceberg query and build | Iceberg | Experimental |
| StarRocks | Iceberg through Arrow Flight SQL | No built-in builder | Experimental |

All backends accept the same `VectorQuery` shape and can return a portable
`pyarrow.Table`. Native terminals keep a query inside the host engine when more
relational work follows: DataFusion returns a DataFrame, Spark returns a
PySpark DataFrame, and StarRocks exposes the generated SQL.

## Query Lifecycle

An indexed query has four logical stages:

1. select the nearest IVF centroids;
2. scan postings for the selected cluster IDs;
3. apply source filters and compute exact squared L2 distances; and
4. order by distance, then retain the requested limit.

The relative order of candidates with equal distance is unspecified.

The backend decides how to execute those stages. Changing backend operators or
optimizations does not change the published index format or query semantics.

## What Relify Is Not

Relify is not a dedicated online vector database. It targets analytical and
batch-oriented vector workloads, including large result sets, similarity
analysis, and vector search composed with relational queries. High-concurrency
serving, replication, admission control, and managed distributed operation are
outside the 0.1 scope.

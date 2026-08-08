# Python API

Relify provides a stable local DataFusion API. Spark Classic and StarRocks
query APIs are bundled under `relify.experimental`. All three consume the same
IVF metadata, backend-independent `VectorQuery`, and shared `ResolvedSearch`
semantics. APIs in the experimental namespace may change without the
compatibility guarantees of the stable API.

For installation and task-oriented workflows, start from the
[documentation index](README.md). This page is the detailed Python behavior
reference.

This document describes implemented behavior. Portable formats and query
semantics are defined by the [`spec/`](../spec/README.md). Production Spark
work and additional engines are tracked in the [`roadmap`](roadmap.md).

## Session

`relify.connect` accepts either a local directory or an explicit SQLite
catalog and index storage root.

```python
import relify

session = relify.connect("./relify-data")
```

The local shortcut creates:

```text
relify-data/
├── catalog.sqlite
├── metadata/
└── indexes/
```

Catalog and index storage can be separated:

```python
session = relify.connect(
    catalog="sqlite:///absolute/path/to/catalog.sqlite",
    index_root="s3://bucket/relify",
    storage_options={
        "aws_region": "us-east-1",
    },
)
```

Index roots support `file`, `s3`, and `hdfs` URIs. The current SQLite catalog
stores both persistent Parquet table definitions and Relify index mappings.
Each session creates exactly one DataFusion context in Rust. Index
construction, source and index I/O, SQL, and DataFrame queries all use that
context. The returned session is that mutable context:

```python
import pyarrow

batch = pyarrow.record_batch([[1, 2]], names=["value"])
session.register_record_batches("weights", [[batch]])
result = session.sql("SELECT * FROM weights").collect()
```

Users may register Arrow or file-backed relations, create views, and execute
DataFusion SQL directly through the session. There is no separate public
context or backend object. Relify registers the stateless Rust
`relify_squared_l2(vector, query_vector)` UDF in the context. Relify searches
normally fuse that distance projection with Top-K into one native batch
operator; the UDF remains available to ordinary DataFusion SQL and as the
fallback for unsupported plan shapes. A session does not require a context
manager or explicit close operation.

The full embedded DataFusion Python API is available from
`relify.datafusion`:

```python
from relify.datafusion import col
from relify.datafusion import functions as functions

df = (
    session.table("weights")
    .filter(col("value") > 0)
    .aggregate(None, [functions.sum(col("value")).alias("total")])
)
```

`relify.datafusion` follows the API and version of the embedded DataFusion
release. It is a separate namespace, so an application may also install the
official `datafusion` package without either package replacing the other.

The local session can also query an Iceberg index built by Spark. Bind the same
PyIceberg catalog name recorded in the index metadata and use the same Relify
index catalog:

```python
from pyiceberg.catalog import load_catalog
import relify

session = relify.connect(
    catalog="sqlite:///data/relify/catalog.sqlite",
    index_root="file:///data/relify/metadata",
    iceberg=load_catalog("lakehouse"),
)
documents = session.table("lakehouse.analytics.documents")
hits = session.to_dataframe(
    documents.search(query_vector, column="embedding").nprobes(128)
)
```

Install Iceberg catalog support with `pip install "relify[iceberg]"`.
DataFusion verifies the Iceberg table UUID and reads the exact source and index
snapshots recorded in Relify metadata. The local builder remains Parquet-only.

### Experimental Spark Session

The experimental Spark integration accepts a caller-owned Spark Classic
session and, for Iceberg sources and construction, a PyIceberg catalog that
points to the same catalog configured in Spark:

```python
from pyiceberg.catalog import load_catalog
import relify

session = relify.experimental.spark.connect(
    spark,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
    iceberg_catalog=load_catalog("lakehouse"),
)
documents = session.table("analytics.documents")
```

Install this path with `pip install "relify[spark]"`. Relify supports Spark
4.0 and 4.1, whose Arrow dependency ranges are compatible with the embedded
DataFusion package and for which Iceberg publishes maintained runtime JARs.
The caller remains responsible for adding the matching Iceberg runtime and
configuring `spark.sql.catalog.lakehouse`.

One Relify Spark session binds one logical Iceberg catalog. The PyIceberg
catalog's name is used in Spark identifiers, so the example resolves
`lakehouse.analytics.documents` through both systems. `session.table` fails
immediately if either side cannot resolve it.

Spark can also query a Parquet index built by the local backend without binding
an Iceberg catalog:

```python
session = relify.experimental.spark.connect(
    spark,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
)
documents = session.register_parquet(
    "documents",
    "s3://bucket/documents/*.parquet",
)
hits = session.to_dataframe(
    documents.search(query_vector, column="embedding").nprobes(128)
)
```

The Parquet source URI must be the same canonical URI recorded by the local
builder. The first Spark builder still requires an Iceberg source and writes
Iceberg index tables.

Spark construction retains the ordinary table API. The Spark session supplies
`relify.experimental.Spark(spark)` as its default builder:

```python
documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=4096),
    writer_options=relify.WriteOptions(partitions=128),
)
documents.wait_for_index("documents_embedding")
```

MLlib trains centroids in Spark. Relify converts them to canonical `float`,
recomputes final assignments in Arrow batches against those exact centroids,
range partitions and locally sorts postings by `(cid, key...)`, uses PyIceberg
to create the canonical required-field schemas, and asks Spark to append the
distributed data to
`lakehouse.relify.documents_embedding_centroids` and
`lakehouse.relify.documents_embedding_postings`. Metadata is published only
after both table snapshots are committed. `WriteOptions.partitions` is optional
and defaults to the active Spark context's parallelism. A large cluster can
span adjacent partitions at key boundaries.

Search returns a native PySpark DataFrame:

```python
query = (
    documents.search(query_vector, column="embedding")
    .where("tenant_id = 42")
    .nprobes(128)
    .limit(10_000)
    .select(["document_id", "title"])
)

hits = session.to_dataframe(query)
result = (
    hits.join(document_stats, "document_id")
    .groupBy("category")
    .count()
)
```

`session.collect(query)` instead executes the Spark plan and returns the common
`pyarrow.Table` result. Use `session.to_dataframe(query).collect()` when native
PySpark rows are required. `session.explain(query, verbose=False)` captures the
public Spark DataFrame plan as text.

Centroid Top-K and postings pruning remain relational operators in the Spark
plan. Relify uses a left-semi join with the selected `cid` DataFrame rather
than expanding a large `IN` list. Source resolution is omitted for an
index-only key/vector projection with no source predicate. Other projections
and predicates join source rows transparently.

The first Spark implementation supports Spark Classic, initial index
construction, native DataFrame queries, and a SQLite development index
catalog. Spark Connect, refresh, cross-driver build coordination, Iceberg
maintenance, remote index catalogs, and production conformance tests remain
open.

### Experimental StarRocks Session

The experimental StarRocks integration is a query session over a caller-owned
Arrow Flight SQL ADBC connection. The same Iceberg catalog must be registered
under the same logical name in StarRocks and PyIceberg:

```python
import adbc_driver_flightsql.dbapi as flight_sql
from adbc_driver_manager import DatabaseOptions
from pyiceberg.catalog import load_catalog
import relify

connection = flight_sql.connect(
    uri="grpc://starrocks.example.com:9408",
    db_kwargs={
        DatabaseOptions.USERNAME.value: "root",
        DatabaseOptions.PASSWORD.value: "",
    },
)
iceberg = load_catalog("lakehouse")

session = relify.experimental.starrocks.connect(
    connection,
    index_catalog="sqlite:///data/relify/catalog.sqlite",
    iceberg_catalog=iceberg,
)
documents = session.table("analytics.documents")
```

Install this path with `pip install "relify[starrocks]"`. It requires
StarRocks 3.5.1 or later because the first implementation uses Arrow Flight
SQL and exact Iceberg `VERSION AS OF` reads.

The normal query builder is unchanged:

```python
query = (
    documents.search(query_vector, column="embedding")
    .where("tenant_id = 42")
    .nprobes(128)
    .limit(10_000)
    .select(["document_id", "title"])
)

hits = session.collect(query)
print(session.explain(query))
```

PyIceberg verifies each source and index table UUID and resolves the schema
active at the referenced snapshot. Relify then compiles centroid Top-K,
postings pruning, source resolution, filtering, exact distance evaluation, and
final Top-K into one StarRocks SQL statement. Every relation is read with its
metadata `snapshot-id`; no candidates are materialized in Python.

The session does not own or close the supplied connection. Each operation uses
one cursor, and `collect` returns one `pyarrow.Table` after validating the
result schema and finite `float32` distances. Empty results retain the expected
schema. `session.to_sql(query)` exposes the generated StarRocks SQL for
inspection.

StarRocks has no implicit index builder. The same table can explicitly use a
caller-owned Spark session for construction:

```python
documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=4096),
    builder=relify.experimental.Spark(spark),
    writer_options=relify.WriteOptions(partitions=128),
)
documents.wait_for_index("documents_embedding")
```

The StarRocks session pins the source Iceberg snapshot and publishes the
resulting metadata; Spark trains and writes the Iceberg index tables. Spark,
StarRocks, and PyIceberg must use the same logical catalog name.

The first implementation supports Iceberg source and index tables only. It
does not create the StarRocks external catalog, build indexes with StarRocks
compute, use StarRocks native vector indexes, query Parquet through `FILES()`,
or provide a StarRocks DataFrame facade. Its SQLite index catalog is a
development catalog; remote coordination remains future work.

The opt-in conformance test loads the shared IVF fixtures into a temporary
Iceberg namespace and queries every case through a real StarRocks deployment.
Its required environment and execution command are documented in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#tests-and-quality-gates).

## Source Tables

Parquet sources use DataFusion's normal registration and lookup flow:

```python
session.register_parquet(
    "documents",
    "file:///absolute/path/to/documents/*/data/*.parquet",
)
documents = session.table("documents")
```

`table(name)` retains DataFusion's lookup semantics and never interprets the
name as a path. A Parquet table registered by the Relify session is returned as
a `SourceTable`, which is a DataFusion `DataFrame` with a stable base-table
binding and index methods. Views and other registered relations remain ordinary
DataFrames. DataFrame transformations such as `select` and `filter` also return
ordinary DataFrames because only a registered base table can own an index.

The catalog persists the fully qualified table identifier, provider, location,
resolved schema, partition columns, pruning configuration, and supported sort
metadata. Reopening the catalog reconstructs the same DataFusion provider, so
`session.table("documents")` works without calling `register_parquet` again.
The identifier cannot be registered to another location until it is explicitly
deregistered.

A location may be one absolute file, a directory, or a `*` wildcard pattern
such as `/data/*/part-*.parquet`. Relify stores the pattern, not the current file
list, and resolves its current matches through the configured `file`, `s3`, or
`hdfs` object store whenever a session reconstructs the table. Registration
options therefore apply consistently to ordinary scans, index construction,
and search. Parquet still has no snapshot semantics: changes to files matched
by the same location remain the caller's consistency responsibility.

## Build an Index

```python
from datetime import timedelta

documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=1024, encoding="lvq8"),
    wait_timeout=timedelta(minutes=5),
)
```

The vector column must be a required `list<float32>` with required, finite
elements of one fixed dimension. Keys may be composite and are copied into the
postings table without creating an internal row identifier. `encoding`
accepts `flat`, `lvq4`, `lvq8`, or `source`. `flat` stores exact vectors, while
LVQ stores compact per-vector codes and evaluates approximate distance directly
from the postings table. `source` stores only keys and resolves candidate
vectors from the source. Omitting `encoding` selects `flat`.

The same table method is implemented by every backend. Local and Spark
sessions expose `session.default_builder` and use it when `builder` is omitted;
StarRocks has no default and requires an explicit compatible builder. On the
local session, the call above starts an asynchronous `Local()` build. Omitting
`wait_timeout` returns after submission. Build state can be inspected or
awaited:

```python
status = documents.index_status("documents_embedding")
documents.wait_for_index(
    "documents_embedding",
    timeout=timedelta(minutes=5),
)
```

`index_status` is source-scoped. Pending and failed local operations come from
the current session; `ready` and `current_snapshot_id` always reflect the
catalog's current mapping. Dropping or re-registering an index is therefore
visible immediately without restarting the session. If a refresh fails while
an earlier snapshot remains published, the state remains `ready` and `error`
reports the failed refresh.

While a local build is pending or running, `phase` identifies the current build
stage. `completed` and `total` report work counters when that stage exposes
them; `total` is zero when it does not. `progress` maps the stages to an
estimated overall fraction in `0.0..=1.0`; it is intended for progress display,
not elapsed-time prediction.

Construction and physical output are configured independently:

```python
builder = relify.Local(
    threads=8,
    max_row_group_rows=None,
    write_batch_rows=8_192,
)
writer_options = relify.WriteOptions(
    partitions=4,
    compression="uncompressed",
    target_file_size=512 * 1024 * 1024,
)
```

When `threads` is `None`, the local builder uses the process's available
parallelism for centroid training. A positive value creates an isolated
centroid-training worker pool for that build.
When `builder.max_row_group_rows` is `None`, postings use
`clamp(ceil(ntotal / nlist), 4_096, 131_072)`. An explicit positive value
overrides the automatic size.

Refresh rebuilds the current source and atomically publishes a new Relify
snapshot:

```python
documents.refresh_index(
    "documents_embedding",
    builder=builder,
    writer_options=writer_options,
    wait_timeout=timedelta(minutes=5),
)
```

## Search

`Table.search` accepts a one-dimensional iterable of numeric values, including
Python lists, tuples, and NumPy arrays. NumPy is optional: Relify recognizes
array-like objects through `ndim` and `tolist()` without importing NumPy. A
two-dimensional array is rejected because batch queries are not yet supported.

An index is selected automatically when exactly one published index matches
the source and vector column:

```python
query = (
    documents.search(query_vector, column="embedding")
    .where("tenant_id = 42 AND status = 'published'")
    .nprobes(16)
    .limit(100)
    .select(["document_id", "title"])
)
hits = session.collect(query)
```

`VectorQuery` is immutable and contains only the structured source-table
identifier and logical search options. The session resolves that identifier
and owns compilation or execution, so the same query value can be submitted to
another session that mounts the same table identifier.

`session.collect(query)` executes immediately and returns one
`pyarrow.Table` on every backend. This portable terminal preserves field types
and nullability for empty results. Use a concrete session's native terminal,
such as `session.to_dataframe(query)`, when the query should remain lazy or
continue through the host engine.

The optional index name disambiguates multiple matching indexes:

```python
hits = session.collect(
    documents.search(
        query_vector,
        column="embedding",
        index="documents_embedding",
    )
)
```

`where` is a prefilter evaluated against source columns before Top-K.
`select` controls returned source columns. `_distance` is always appended.

Exact search is available without an index:

```python
query = (
    documents.search(query_vector, column="embedding")
    .bypass_vector_index()
    .limit(100)
)
hits = session.collect(query)
```

Query planning and runtime analysis follow the same query-builder shape:

```python
query = (
    documents.search(query_vector, column="embedding")
    .where("tenant_id = 42 AND status = 'published'")
    .nprobes(16)
    .limit(100)
)

print(session.explain(query))
print(session.explain(query, verbose=True))
print(session.analyze(query))
```

`session.explain(query, verbose=False)` returns the DataFusion plan without
executing the source scan or final Top-K. Resolving catalog metadata and
selecting IVF clusters are planning work. `session.analyze(query)` executes the
query once and returns the physical plan annotated with runtime metrics such as
output rows, elapsed time, bytes scanned, distance and candidate-selection time,
candidate counts, retained batch memory, final candidate-sort and projection
time, and rows rejected by the dynamic Top-K threshold.

Both methods return diagnostic `str` values, not a stable or machine-readable
plan type. In-place replacement of Parquet source contents remains outside the
reader's consistency guarantees; query planning does not perform full-table
cardinality or key-uniqueness scans.

## Backend Discovery and Capabilities

Ordinary users should keep using the concrete integration entry points above.
Configuration-driven applications can discover installed integrations without
importing their modules:

```python
for backend in relify.backends.installed():
    print(backend.name, backend.distribution, backend.version)

plugin = relify.backends.load("local")
assert plugin.info.name == "local"
```

The bundled experimental Spark and StarRocks integrations are intentionally
absent from this stable registry.

Every session exposes `session.backend` and a typed
`session.capabilities` report. The plugin declaration is the implementation's
upper bound; the session's available subset reflects bound catalogs and other
runtime configuration. Query profiles name explicit index-family,
source-profile, and index-profile combinations. Native terminals and
maintenance operations are reported separately.

Third-party integration packaging, entry points, shared planning values, and
contract tests are documented in [`backends.md`](backends.md).

## Query Composition

`session.to_dataframe(query)` exposes the search as a native lazy DataFusion
relation in the session's single context:

```python
from relify.datafusion import col, functions

query = documents.search(query_vector).limit(100)
hits = session.to_dataframe(query)
document_stats = session.read_parquet(
    "file:///data/document_stats.parquet"
)
result = (
    hits.join(document_stats, on="document_id")
    .aggregate(
        "category",
        [
            functions.count(col("document_id")).alias("matches"),
            functions.avg(col("_distance")).alias("avg_distance"),
        ],
    )
    .sort("category")
    .collect()
)
```

The search, join, and aggregation remain one lazy DataFusion plan. The combined
plan can use ordinary DataFusion optimizations such as projection and predicate
pushdown, join-side selection and reordering where legal, repartitioning, and
runtime filters supported by the selected join strategy.

`session.to_sql(query)` returns the complete search SQL over the source table
and immutable index relations registered in the same session:

```python
search_sql = session.to_sql(query)
result = session.sql(
    f"""
    SELECT *
    FROM ({search_sql}) AS vector_hits
    WHERE _distance >= 0
    """
).collect()
```

The SQL refers to the source's existing DataFusion registration and stable
session-internal names for immutable index relations. Relify does not register
the whole search as a hidden temporary view. The returned SQL is therefore
executable by the originating session, not portable standalone SQL. Neither a
lazy DataFrame nor the SQL string creates persistent reader coordination;
garbage-collection retention is the reader-safety boundary. These operations
do not change the IVF prefilter or Top-K semantics. When the SQL is planned,
each internal index relation uses the current session cache if present and
otherwise reads immutable Parquet; the stable name itself does not pin cached
Arrow data.

The session can explicitly materialize a complete current index snapshot before
a repeated-query workload:

```python
cached = session.cache_index("documents_embedding")
assert session.is_index_cached("documents_embedding")

# Search automatically uses the cached index relations.
hits = session.collect(documents.search(query_vector, column="embedding"))

session.uncache_index("documents_embedding")
```

`cache_index` blocks until every index relation is decoded into Arrow memory
and returns its snapshot ID, relation count, and approximate resident bytes.
It does not cache the source table. Refreshing, registering, or dropping the
index invalidates the cached snapshot.

## Catalog

The Python catalog facade exposes the current root namespace:

```python
names = session.indexes.list()
entry = session.indexes.load("documents_embedding")
source = entry.metadata["snapshots"][0]["source"]
matching = session.indexes.list_for(source)
selected = session.indexes.select(source, column="embedding")
session.indexes.register("recovered_index", entry.metadata_location)
session.indexes.drop("recovered_index")
```

`drop` removes catalog visibility without deleting metadata or index tables.
`register` imports an existing metadata file managed below the session's index
root. `list_for` and `select` use an exact portable relation reference and are
the query-engine-independent discovery surface used by backend integrations.

A query-only integration may open the same facade without constructing an
execution session:

```python
indexes = relify.open_index_catalog(
    "sqlite:///data/relify/catalog.sqlite",
    metadata_root="s3://indexes/metadata",
    storage_options={"aws_region": "us-east-1"},
)
```

Source-centered discovery and removal are available on every Relify table:

```python
indexes = documents.list_indexes()
documents.drop_index("documents_embedding")
```

## Maintenance

Unreachable managed objects are retained unless they lost catalog reachability
before the caller-supplied cutoff and satisfy Relify's minimum seven-day
retention rule:

```python
from datetime import UTC, datetime, timedelta

candidates = session.maintenance.remove_orphans(
    older_than=datetime.now(UTC) - timedelta(days=7),
)
```

The operation is a dry run by default. Deletion must be requested explicitly:

```python
removed = session.maintenance.remove_orphans(
    older_than=datetime.now(UTC) - timedelta(days=7),
    dry_run=False,
)
```

For published objects, the cutoff applies to the time their metadata document
lost catalog reachability, not file creation time. Unknown files that may come
from an unfinished operation are retained for at least seven days. There is no
public option to bypass this safety window. A query that outlives the retention
period after its selected snapshot becomes unreachable is not guaranteed to
survive collection.

## Current Limits

See [current limitations](limitations.md) for the complete release and
operational boundary. At the Python API level, the local implementation
supports:

- one SQLite catalog for Parquet definitions and index mappings;
- the root index namespace in the Python facade;
- Parquet source and index tables;
- `source`, `flat`, `lvq4`, and `lvq8` IVF postings with squared L2 distance;
- one local Rust builder; and
- one native DataFusion session for build, query, and relational composition.

The experimental Spark implementation supports one Iceberg catalog, initial
IVF construction, and native DataFrame queries on Spark Classic. The
experimental StarRocks implementation supports query-only Iceberg reads over
Arrow Flight SQL.
Portable standalone SQL compilation, Spark Connect, remote index catalogs,
StarRocks construction, and StarRocks Parquet reads are not implemented.

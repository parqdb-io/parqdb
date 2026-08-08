# Local DataFusion and Parquet

The local backend runs in the Python process and is the stable Relify 0.1
path. It uses DataFusion for relational execution, native Rust code for index
construction and distance kernels, SQLite for catalog state, and Parquet for
the index relations.

Use it for local development, single-node analytical workloads, batch jobs, or
applications that already expose Parquet data to an embedded query engine.

## Install

```bash
python -m pip install relify
```

See [getting started](../getting-started.md) for the supported Python and
platform matrix and a complete first query.

## Open a Session

The shortest configuration keeps the catalog and index data together:

```python
import relify

session = relify.connect("./relify-data")
```

The directory contains the SQLite catalog, immutable metadata documents, and
Parquet index snapshots. Reopening the same path restores persistent source
registrations and index mappings.

Separate the local catalog from index storage when the relations belong on
shared storage:

```python
session = relify.connect(
    catalog="sqlite:///var/lib/relify/catalog.sqlite",
    index_root="s3://lakehouse-indexes/relify",
    storage_options={"aws_region": "us-east-1"},
)
```

The SQLite path must be absolute. See [configuration](../configuration.md) for
S3-compatible storage and HDFS settings.

## Register a Source

Register one absolute file, directory, or wildcard pattern:

```python
session.register_parquet(
    "documents",
    "file:///data/documents/*/part-*.parquet",
)
documents = session.table("documents")
```

The registration is durable. A new session opened on the same catalog can call
`session.table("documents")` without registering it again. Relify stores the
location pattern rather than a one-time list of matching files.

Use `session.deregister_table("documents")` before rebinding the same logical
name to a different location.

## Build and Monitor an Index

```python
from datetime import timedelta

documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=4096),
)
documents.wait_for_index(
    "documents_embedding",
    timeout=timedelta(minutes=30),
)
```

Construction runs asynchronously. Inspect progress without blocking:

```python
status = documents.index_status("documents_embedding")
print(status.state, status.phase, status.completed, status.total)
```

Tune CPU and physical Parquet output independently:

```python
documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=4096, encoding="lvq8"),
    builder=relify.Local(threads=8),
    writer_options=relify.WriteOptions(
        compression="zstd(3)",
        target_file_size=512 * 1024 * 1024,
    ),
)
```

Start with defaults. `nlist` and query-time `nprobes` have the largest effect
on IVF recall and candidate work. Writer settings primarily affect persistence
and scan layout.

## Query and Continue with DataFusion

Collect a portable Arrow result:

```python
query = (
    documents.search(query_vector, column="embedding")
    .where("tenant_id = 42")
    .nprobes(64)
    .limit(1_000)
    .select(["document_id", "category"])
)
hits = session.to_arrow(query)
```

Keep the search lazy when additional relational work follows:

```python
from relify.datafusion import col, functions

hits = session.to_dataframe(query)
result = (
    hits.aggregate(
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

The vector search and aggregation remain in one DataFusion plan. Use
`session.to_sql(query)` when SQL composition is more convenient; the returned
SQL is executable in the originating session because it refers to registered
source and index relations.

## Cache Repeated Queries

Materialize one immutable index snapshot as decoded Arrow data before a
repeated-query workload:

```python
cached = session.cache_index("documents_embedding")
print(cached.snapshot_id, cached.resident_bytes)

assert session.is_index_cached("documents_embedding")
session.uncache_index("documents_embedding")
```

The cache includes index relations, not the source table. A refresh, catalog
replacement, or drop invalidates the cached snapshot.

## Refresh and Remove

After the source changes, rebuild and atomically publish a new snapshot:

```python
documents.refresh_index("documents_embedding")
documents.wait_for_index("documents_embedding")
```

Dropping an index removes catalog visibility but deliberately leaves immutable
objects for retention-safe cleanup:

```python
documents.drop_index("documents_embedding")
```

Use the maintenance API only after reading the retention contract in the
[Python API](../python-api.md#maintenance).

## Diagnose a Query

```python
print(session.explain(query))
print(session.analyze(query))
```

`analyze` reports candidate counts, distance and Top-K time, bytes scanned, and
other physical metrics. See [troubleshooting](../troubleshooting.md) for common
schema, catalog, storage, and performance failures.

# Local DataFusion and Parquet

The embedded runtime runs in the Python process and is the stable Relify 0.1
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

Store index relations on shared storage while keeping the local session state
under one directory:

```python
session = relify.connect(
    "/var/lib/relify",
    warehouse="s3://lakehouse-indexes/relify",
    storage_options={"aws_region": "us-east-1"},
)
```

See [configuration](../configuration.md) for S3-compatible storage and HDFS
settings.

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

Tune physical Parquet output independently:

```python
documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=4096, encoding="lvq8"),
    writer_options=relify.WriteOptions(
        compression="zstd(3)",
        target_file_size=512 * 1024 * 1024,
    ),
)
```

Start with defaults. `nlist` and query-time `nprobes` have the largest effect
on IVF recall and candidate work. Writer settings primarily affect persistence
and scan layout.

The default `source` encoding keeps vectors in the source table. `lvq4` and
`lvq8` store compact codes in postings. Set `metric="cosine"` to normalize
source and query vectors and report cosine distance; the default is
`l2_squared`.

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

Compile the vector query as a SQL subquery when relational work follows:

```python
search_sql = session.to_sql(query)
result = session.sql(f"""
    SELECT category, COUNT(*) AS matches, AVG(_distance) AS avg_distance
    FROM ({search_sql}) AS hits
    GROUP BY category
    ORDER BY category
""")
```

The vector search and aggregation remain in one DataFusion plan. The generated
SQL is executable in the originating session because it refers to registered
source and index relations.

## Page Cache

Repeated queries use the session's bounded decompressed Parquet Page cache:

```python
stats = session.parquet_page_cache_stats()
print(stats.resident_bytes, stats.hits, stats.misses)
session.clear_parquet_page_cache()
```

The cache is capacity-bounded and shared by Parquet index and source scans in
the session. Configure it with `relify.parquet.page_cache.capacity`.

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

Retention and orphan cleanup behavior is summarized in
[current limitations](../limitations.md).

## Diagnose a Query

```python
print(session.explain(query))
print(session.analyze(query))
```

`analyze` reports candidate counts, distance and Top-K time, bytes scanned, and
other physical metrics. See [troubleshooting](../troubleshooting.md) for common
schema, catalog, storage, and performance failures.

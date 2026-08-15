# Python API

Relify exposes synchronous and asynchronous facades over one service contract.
The current transport is embedded and executes through bundled DataFusion.

## Connect

```python
import relify

session = relify.connect("./relify-data")
```

The path contains the SQLite catalog and, by default, the index warehouse.
Use a context manager when session lifetime is scoped:

```python
with relify.connect("./relify-data") as session:
    print(session.list_tables())
```

An explicit catalog and index root may be configured independently:

```python
session = relify.connect(
    catalog="sqlite:///absolute/path/catalog.sqlite",
    index_root="s3://bucket/relify/",
    storage_options={"aws_region": "us-east-1"},
)
```

`Session` is not a DataFusion `SessionContext`. Call
`session.datafusion_context()` only when an embedded-only DataFusion operation
is intentionally required.

## Register and Discover Tables

```python
session.register_parquet("documents", "s3://bucket/documents/*.parquet")

identifiers = session.list_tables()
documents = session.table("documents")
print(documents.identifier)
print(documents.schema)

session.deregister_table("documents")
```

Registration persists the source definition in the session catalog. It does
not copy source rows. Reopening the catalog reconstructs the Parquet provider.
A registered identifier cannot be silently rebound to a different source.

`register_parquet` also accepts DataFusion Parquet options such as an explicit
Arrow schema, partition columns, pruning, file extension, and sort order.

## Build an Index

```python
from datetime import timedelta

documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(
        nlist=4096,
        encoding="lvq8",
        metric="cosine",
    ),
    writer_options=relify.WriteOptions(
        partitions=32,
        compression="zstd(3)",
        target_file_size=512 * 1024 * 1024,
    ),
)
documents.wait_for_index(
    "documents_embedding",
    timeout=timedelta(hours=1),
)
```

`create_index` submits work to the native, process-scoped build coordinator.
The client does not pass executable builder objects. Omitting `wait_timeout`
returns after submission; passing it combines submission and waiting in one
call. Accepted work is independent of the waiting client, but does not survive
a process restart.

Supported `IVF.encoding` values are:

- `source`: postings contain source keys and exact distance resolves vectors
  from the source table;
- `lvq8`: postings contain one 8-bit locally quantized code per dimension; and
- `lvq4`: postings pack two 4-bit codes per byte.

Supported metrics are `l2_squared` and `cosine`. Cosine construction and query
normalize vectors before using squared Euclidean ranking.

`WriteOptions` controls physical Parquet output:

| Field | Default | Meaning |
| --- | --- | --- |
| `partitions` | automatic | Concurrent output partitions |
| `compression` | `uncompressed` | Parquet compression codec |
| `target_file_size` | 512 MiB | Target output file size |
| `max_row_group_rows` | automatic | Optional row-group row limit |
| `write_batch_rows` | 8192 | Writer input batch size |

## Index Lifecycle

```python
status = documents.index_status("documents_embedding")
print(status.state, status.progress, status.phase)

for index in documents.list_indexes():
    print(index.name, index.current_snapshot_id)

documents.refresh_index(
    "documents_embedding",
    config=relify.IVF(nlist=8192, encoding="lvq8", metric="cosine"),
)
documents.wait_for_index("documents_embedding")
documents.drop_index("documents_embedding")
```

Refresh builds a replacement and atomically publishes it. The previous
snapshot remains queryable until the replacement is complete.

## Build a Vector Query

```python
query = (
    documents.search(vector, column="embedding")
    .where("tenant_id = 42 AND status = 'published'")
    .nprobes(64)
    .limit(100)
    .select(["document_id", "title"])
)
```

`VectorQuery` is immutable. Each modifier returns a new value. Use `index=` in
`search` to select among multiple indexes on one vector field. Use
`bypass_vector_index()` for the exact reference path.

The result contains selected source columns followed by required query columns,
including `_distance`.

## Execute and Stream

```python
hits = session.collect(query)       # pyarrow.Table
same = session.to_arrow(query)      # compatibility alias

with session.stream(query) as reader:
    for batch in reader:
        consume(batch)
```

`stream` returns a real `pyarrow.RecordBatchReader`. Closing the reader cancels
unfinished execution and releases its query admission slot.

SQL strings use the same terminals:

```python
summary = session.sql("""
    SELECT category, COUNT(*) AS matches
    FROM documents
    GROUP BY category
""")

search_sql = session.to_sql(query)
combined = session.sql(f"""
    SELECT category, COUNT(*) AS matches, MIN(_distance) AS nearest
    FROM ({search_sql}) AS hits
    GROUP BY category
""")
```

Inspect vector or SQL execution with:

```python
print(session.explain(query))
print(session.explain(query, verbose=True))
print(session.analyze(query))
```

## Asynchronous API

```python
session = await relify.connect_async("./relify-data")
try:
    documents = await session.table("documents")
    query = documents.search(vector, column="embedding").limit(10)

    async for batch in await session.stream(query):
        consume(batch)
finally:
    await session.close()
```

Catalog access, index lifecycle, execution, and stream consumption are
awaitable. Query construction remains synchronous because it only creates an
immutable value. The synchronous facade invokes this same asynchronous service
path through one long-lived blocking bridge.

## Session Configuration

Relify extends the bundled DataFusion configuration:

```python
config = (
    relify.SessionConfig()
    .set("relify.execution.query_dop", "8")
    .set("relify.execution.query_concurrency", "16")
    .set("relify.execution.query_queue_capacity", "64")
    .set("relify.parquet.page_cache.capacity", "4294967296")
)

session = relify.connect("./relify-data", config=config)
```

Runtime resource configuration is resolved when the embedded session is
created. See [configuration](configuration.md) for the complete set of keys.

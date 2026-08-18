# Python API

ParqDB exposes synchronous and asynchronous facades over one service contract.
It selects embedded execution or the HTTP transport from the connection
location.

## Connect

```python
import parqdb

session = parqdb.connect("./parqdb-data")
```

The path contains the SQLite catalog and, by default, the index warehouse.
Use a context manager when session lifetime is scoped:

```python
with parqdb.connect("./parqdb-data") as session:
    print(session.list_tables())
```

Index relations may be stored in a separate warehouse while catalog state stays
under the local root:

```python
session = parqdb.connect(
    "/var/lib/parqdb",
    warehouse="s3://bucket/parqdb/",
    storage_options={"aws_region": "us-east-1"},
)
```

`Session` is not a DataFusion `SessionContext`. Call
`session.datafusion_context()` only when an embedded-only DataFusion operation
is intentionally required.

### Connect to a ParqDB Server

Install ParqDB in the server environment:

```bash
python -m pip install parqdb
```

Start the server with its default TOML configuration:

```bash
parqdb config init
parqdb serve
```

See the [server guide](guides/server.md) for source authorization, storage,
runtime settings, and the ASGI embedding API.

```python
session = parqdb.connect("http://127.0.0.1:8000")
documents = session.table("documents")
hits = session.collect(documents.search(vector).limit(10))
```

The same table and index lifecycle API is available remotely. Paths passed to
`register_parquet` are resolved by the server; the operation does not upload
client files or credentials. Index builds are accepted asynchronously, and
`wait_for_index` polls their server-side status. `datafusion_context()`
remains embedded-only.

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
    config=parqdb.IVF(
        nlist=4096,
        encoding="lvq8",
        metric="cosine",
    ),
    writer_options=parqdb.WriteOptions(
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

## Register a Published Index

An existing index can be attached without rebuilding or copying data when its
metadata and Parquet relations are already present in the session warehouse:

```python
documents.register_index(
    "documents_embedding",
    metadata_location=(
        "s3://shared-parqdb/metadata/"
        "2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1/v1.metadata.json"
    ),
)
```

The selected table supplies the source binding. Registration validates the
source schema and row count, centroid metadata, postings schema, and every
warehouse-relative location before atomically creating the catalog entry. A
session has one warehouse; there is no per-index root parameter.

## Index Lifecycle

```python
status = documents.index_status("documents_embedding")
print(status.state, status.progress, status.phase)

for index in documents.list_indexes():
    print(index.name, index.current_snapshot_id)

documents.refresh_index(
    "documents_embedding",
    config=parqdb.IVF(nlist=8192, encoding="lvq8", metric="cosine"),
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
session = await parqdb.connect_async("./parqdb-data")
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

ParqDB extends the bundled DataFusion configuration:

```python
config = (
    parqdb.SessionConfig()
    .set("parqdb.execution.query_dop", "8")
    .set("parqdb.execution.query_concurrency", "16")
    .set("parqdb.execution.query_queue_capacity", "64")
    .set("parqdb.parquet.page_cache.capacity", "4294967296")
)

session = parqdb.connect("./parqdb-data", config=config)
```

Runtime resource configuration is resolved when the embedded session is
created. See [configuration](configuration.md) for the complete set of keys.

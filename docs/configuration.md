# Configuration

## Installation

The base package contains the embedded DataFusion runtime, native Rust
extension, Parquet support, and SQLite catalog:

```bash
python -m pip install relify
```

Install PyIceberg support when exact Iceberg table references are required:

```bash
python -m pip install "relify[iceberg]"
```

## Connection

The compact local form creates `catalog.sqlite` and the index warehouse under
one directory:

```python
session = relify.connect("./relify-data")
```

Catalog and index storage may be configured separately:

```python
session = relify.connect(
    catalog="sqlite:///absolute/path/catalog.sqlite",
    index_root="s3://bucket/relify/",
    storage_options={
        "aws_region": "us-east-1",
        "aws_endpoint": "https://s3.example.com",
    },
)
```

`root` and `catalog` are mutually exclusive. An explicit catalog requires an
explicit `index_root`. The first implementation supports SQLite catalog URIs.

Storage locations must be absolute canonical `file`, `s3`, or `hdfs` URIs.
Credentials are process configuration and are never written into index
metadata.

## Server Source Policy

An HTTP server rejects source registration unless the canonical source is below
an explicitly allowed server-visible prefix:

```python
from relify.server import create_app

app = create_app(
    "/srv/relify",
    allowed_source_prefixes=["/srv/lakehouse", "s3://bucket/documents"],
)
```

The default allowlist is empty. File paths are resolved by the server and must
remain below an allowed file root. Object-store URIs must match the configured
scheme, authority, and path-segment boundary. Registration requests cannot
override the server's storage credentials or endpoint configuration.

## Session Configuration

Use `relify.SessionConfig`; it extends the bundled DataFusion configuration:

```python
config = (
    relify.SessionConfig()
    .set("relify.execution.query_dop", "8")
    .set("relify.execution.query_concurrency", "16")
    .set("relify.execution.query_queue_capacity", "64")
    .set("relify.execution.query_queue_timeout", "5s")
    .set("relify.build.dop", "8")
)

session = relify.connect("./relify-data", config=config)
```

| Key | Meaning |
| --- | --- |
| `relify.execution.query_dop` | DataFusion partitions available to one query |
| `relify.execution.query_concurrency` | Active query admission slots |
| `relify.execution.query_queue_capacity` | Maximum queued queries |
| `relify.execution.query_queue_timeout` | Maximum queue wait |
| `relify.build.dop` | Worker count used by an accepted index build |

Resource settings are resolved when the session is created. Changing a
DataFusion `SET` value later does not rebuild the process runtime.

## Cache Configuration

Relify uses bounded caches for immutable metadata, index planning state, and
decompressed Parquet pages:

```python
config = (
    relify.SessionConfig()
    .set("relify.metadata.cache.max_entries", "1024")
    .set("relify.metadata.cache.max_bytes", "268435456")
    .set("relify.query.manifest.cache.max_entries", "256")
    .set("relify.query.manifest.cache.max_bytes", "2147483648")
    .set("relify.query.centroid.cache.max_entries", "256")
    .set("relify.query.centroid.cache.max_bytes", "2147483648")
    .set("relify.parquet.page_cache.capacity", "4294967296")
)
```

A zero capacity disables the corresponding cache. Immutable metadata locations
and file identity are the consistency keys; publishing a new snapshot does not
mutate an existing cache entry.

## Parquet Registration

```python
session.register_parquet(
    "documents",
    "s3://bucket/documents/*.parquet",
    parquet_pruning=True,
    file_extension=".parquet",
)
```

Persistent registration currently requires one path or wildcard pattern.
Optional partition columns, Arrow schema, and sort order follow the bundled
DataFusion API.

## Index Configuration

```python
config = relify.IVF(
    nlist=4096,
    encoding="lvq8",
    metric="cosine",
)
```

`encoding` accepts `source`, `lvq4`, or `lvq8`. `metric` accepts
`l2_squared` or `cosine`.

Physical Parquet output is configured separately:

```python
options = relify.WriteOptions(
    partitions=32,
    compression="zstd(3)",
    target_file_size=512 * 1024 * 1024,
    max_row_group_rows=65_536,
    write_batch_rows=8_192,
)
```

Build implementation and worker ownership are deployment concerns. The public
API does not accept a Python builder object. A native `LocalSession` accepts
builds into one process-scoped queue and runs one build at a time. Accepted
work survives client cancellation or disconnect, but not process restart.

## DataFusion Escape Hatch

The portable session does not proxy arbitrary DataFusion methods. Embedded
applications may explicitly obtain the bundled context:

```python
context = session.datafusion_context()
context.register_udf(...)
```

Objects registered only through this context are process-local and are outside
client/server portability guarantees.

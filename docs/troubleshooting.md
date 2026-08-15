# Troubleshooting

Start with the smallest failing operation: import the package, open the
catalog, resolve the source, list the index, explain the query, and only then
execute it. Include the failing step and its exact configuration when opening
an issue.

## Installation

### No compatible wheel is available

Confirm all three values:

```bash
python -VV
python -c "import platform; print(platform.system(), platform.machine())"
python -c "import sysconfig; print(sysconfig.get_platform())"
```

Relify 0.1 publishes wheels for CPython 3.11 through 3.14 on Linux x86_64 with
glibc 2.28 or later and macOS arm64 11 or later. Free-threaded interpreters and
other platforms are outside the initial wheel matrix.

### An optional backend cannot be imported

Install the matching extra in the same interpreter environment:

```bash
python -m pip install "relify[spark]"
python -m pip install "relify[starrocks]"
python -m pip show relify
```

The base package intentionally does not install PySpark, PyIceberg, or the
Flight SQL driver unless selected.

## Source Registration

### A local path changes identity between runs

Use an absolute path or canonical `file:///` URI. Relify resolves local paths
when they are registered and persists that source identity in SQLite.

### A wildcard matches no files

Relify supports `*` within path segments, but a wildcard does not cross `/`.
Verify the URI authority, prefix, file extension, and object-store list
permissions. Registration stores the pattern and resolves it again when a
session reopens.

### The vector schema is rejected

For a local file, inspect the physical Parquet schema:

```python
import pyarrow.parquet as parquet

print(parquet.ParquetFile("/data/documents.parquet").schema_arrow)
```

Every vector value must be non-null, have one fixed dimension, and contain only
non-null finite `float32` or `float64` elements. The physical source schema may
declare list elements nullable, but indexed values cannot contain nulls. The
source and postings key types must match exactly. Use the
[IVF schema specification](../spec/ivf/index-schema.md) as the normative
contract. A compute engine may expose a more conservatively nullable query
schema; that query schema does not replace the file or table schema used for
conformance.

## Catalog and Index Selection

### The index disappears in another process

Confirm both processes use the same absolute SQLite catalog path and metadata
root. `relify.connect("./relify-data")` resolves relative to each process's
working directory, so use an explicit absolute path in deployed jobs.

### No index matches the query

Inspect the source-scoped catalog entries:

```python
print(documents.list_indexes())
print(session.indexes.list())
```

An index matches an exact source relation and vector column. Registering the
same files under a different logical source or URI does not silently rebind an
existing index.

### More than one index matches

Specify the index name explicitly:

```python
documents.search(
    query_vector,
    column="embedding",
    index="documents_embedding",
)
```

## Index Construction

### A build appears stuck

Read its structured status before interrupting it:

```python
status = documents.index_status("documents_embedding")
print(status)
```

`phase`, `completed`, and `total` distinguish training, assignment, writing,
and publication. A zero `total` means that phase does not expose a work total;
it does not by itself indicate a deadlock.

### A build fails but the old index still appears ready

Refresh publication is atomic. If a refresh fails, the previous snapshot
remains current while `index_status(...).error` reports the failed operation.

### Index files remain after drop

Dropping removes catalog visibility immediately and retains immutable objects
for safe garbage collection. Preview and then execute orphan removal only
after the minimum seven-day retention interval described in the
[maintenance reference](python-api.md#maintenance).

## Query Results and Performance

### Distances look larger than expected

`_distance` is squared Euclidean distance, not its square root. Ranking is the
same as Euclidean distance, but the numeric value differs.

### Recall is lower than expected

Increase `nprobes`, verify that query and source vectors use the same embedding
model and normalization, and compare with
`.bypass_vector_index()` on a representative query set. `nprobes` cannot
exceed the index's `nlist`.

### A repeated query spends time scanning or decoding the index

Inspect whether the bounded Parquet Page cache is serving the workload:

```python
before = session.parquet_page_cache_stats()
session.collect(query)
after = session.parquet_page_cache_stats()
print(after.hits - before.hits, after.misses - before.misses)
```

If misses remain high, check `relify.parquet.page_cache.capacity`, index pruning,
and whether the workload reuses the same index partitions.

### Determine where time is spent

```python
print(session.explain(query))
print(session.analyze(query))
```

Use candidate counts, bytes scanned, distance time, and final Top-K time before
changing index or writer configuration.

## S3 and HDFS

### S3 credentials or endpoints are ignored

All explicit `storage_options` keys and values must be strings. For an HTTP
S3-compatible service, set `aws_endpoint`, `aws_allow_http="true"`, and usually
`aws_virtual_hosted_style_request="false"`. Check bucket list, read, write, and
delete permissions separately.

### HDFS cannot resolve or connect to the NameNode

Use an absolute `hdfs://authority/path` URI. Validate Hadoop configuration,
authentication, DNS, and network access from the Python process. Relify passes
the supplied string options to its native HDFS client but does not deploy or
discover the cluster.

## Spark and Iceberg

### Spark cannot load an Iceberg table

Verify that the Spark session includes the Iceberg runtime matching its Spark
and Scala versions and that `spark.sql.catalog.<name>` is configured. Relify
supports Spark Classic 4.0 and 4.1, not Spark Connect.

### Spark resolves a table but PyIceberg does not

The two clients must point to the same catalog and warehouse under the same
logical name. Load the table independently through both APIs before opening a
Relify session. Relify uses PyIceberg for UUID, schema, metadata location, and
snapshot validation.

### Spark index construction is unavailable

The experimental Spark integration is query-only for the shared-IVF schema.
Build the current Parquet format with the local backend or register an index
published by a conforming external builder.

## StarRocks and Iceberg

### The Flight SQL connection fails

Test the caller-owned ADBC connection before constructing the Relify session:

```python
cursor = connection.cursor()
cursor.execute("SELECT 1")
print(cursor.fetchall())
cursor.close()
```

Check the Flight SQL port, TLS scheme, username, password, and network path.
The MySQL protocol port is not an Arrow Flight SQL endpoint.

### StarRocks cannot resolve a relation

The `catalog_name` passed to Relify must match the external Iceberg catalog in
StarRocks and the logical PyIceberg catalog name. StarRocks must support exact
`VERSION AS OF` reads for the snapshots recorded by the index.

### Inspect generated SQL

```python
print(session.to_sql(query))
print(session.explain(query))
```

Run the generated statement through the same connection to distinguish SQL or
catalog failures from Relify result validation.

## Reporting an Issue

Include:

- `python -VV` and `python -m pip show relify`;
- operating system and architecture;
- backend and engine versions;
- source, catalog, and index URI schemes with secrets removed;
- `session.explain(query)` output when planning succeeds; and
- the smallest reproducible schema, index configuration, and query.

Use GitHub issues for bugs and feature requests. Report vulnerabilities through
the private process in the [security policy](../SECURITY.md).

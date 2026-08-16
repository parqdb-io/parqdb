# Troubleshooting

Reduce a failure to the smallest operation: import, connect, register, resolve
the table, list the index, explain the query, then execute it.

## Installation

Check the interpreter and platform when no wheel is available:

```bash
python -VV
python -c "import platform; print(platform.system(), platform.machine())"
python -c "import sysconfig; print(sysconfig.get_platform())"
```

Relify publishes CPython 3.11 through 3.14 wheels for Linux x86_64 with glibc
2.28 or later and macOS arm64 11 or later.

## Source Registration

Use an absolute path or canonical URI in deployed jobs. Relative paths are
resolved when registered and that identity is persisted in SQLite.

If a wildcard matches no files, verify its authority, prefix, extension, and
object-store list permissions. `*` does not cross a `/` boundary.

Inspect rejected vector schemas at the physical Parquet layer:

```python
import pyarrow.parquet as pq
print(pq.ParquetFile("/data/documents.parquet").schema_arrow)
```

Vectors must be non-null, finite, and one fixed dimension. Key columns must use
supported exact-equality types and contain no null values.

## Catalog and Index Selection

Two processes must use the same local Relify root and index warehouse. List
indexes through the source table:

```python
print(documents.list_indexes())
```

An index is scoped to an exact registered source and vector column. Specify
`index=` in `documents.search(...)` when more than one index matches.

## Index Construction

Inspect structured progress before interrupting a build:

```python
status = documents.index_status("documents_embedding")
print(status.state, status.phase, status.completed, status.total, status.error)
```

A failed refresh leaves the previous published snapshot current. Dropping an
index removes catalog visibility immediately but leaves immutable objects until
the retention-safe orphan collector removes them.

## Query Results and Performance

`_distance` is squared Euclidean distance for `l2_squared`; it is not square
rooted. Increase `nprobes` when recall is insufficient and compare against
`.bypass_vector_index()` on representative queries.

Use plans and runtime metrics before changing layout:

```python
print(session.explain(query))
print(session.analyze(query))
```

Repeated queries should benefit from the bounded Parquet page cache. If misses
remain high, verify cluster pruning and increase
`relify.parquet.page_cache.capacity` at session creation.

## S3 and HDFS

All explicit `storage_options` keys and values must be strings. S3-compatible
HTTP endpoints usually require `aws_endpoint`, `aws_allow_http="true"`, and
`aws_virtual_hosted_style_request="false"`.

HDFS locations must be absolute `hdfs://authority/path` URIs. Relify does not
deploy the NameNode or discover Hadoop configuration for the application.

## Reporting an Issue

Include Python and Relify versions, operating system and architecture, URI
schemes with secrets removed, `session.explain(query)` when planning succeeds,
and the smallest reproducible schema and query. Report vulnerabilities through
the private process in the [security policy](../SECURITY.md).

# Getting Started

This guide installs Relify, builds an IVF index over a Parquet source, and
runs a filtered vector query through the embedded DataFusion backend. It uses
the small dataset included in the wheel, so no service or external data is
required.

## Requirements

Relify 0.1 supports standard CPython 3.11 through 3.14 on:

- Linux x86_64 with glibc 2.28 or later; and
- macOS arm64 11 or later.

Free-threaded Python and other operating-system or architecture combinations
are outside the initial binary release scope.

## Install

Install the local DataFusion and Parquet path:

```bash
python -m pip install relify
```

With uv:

```bash
uv add relify
```

Optional integrations are installed separately:

```bash
python -m pip install "relify[iceberg]"
python -m pip install "relify[spark]"
python -m pip install "relify[starrocks]"
```

The Spark and StarRocks packages provide client-side dependencies. They do not
deploy either engine or configure an Iceberg catalog.

Use `python -m pip install --pre relify` when explicitly opting into a future
pre-release while a stable release is also available.

## Build an Index

Create `quickstart.py`:

```python
import relify

session = relify.connect("./relify-data")
source = relify.datasets.uri("documents")

if not session.table_exist("documents"):
    session.register_parquet("documents", source)
documents = session.table("documents")

if "documents_embedding" not in session.indexes.list():
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["document_id"],
        config=relify.IVF(nlist=3),
    )
    documents.wait_for_index("documents_embedding")
```

The source table remains in its original Parquet dataset. Relify writes the
index and its metadata below `./relify-data`; it does not copy the source rows
into another database.

## Search

Append a filtered search to the same file:

```python
query = (
    documents.search([0.2, 0.0], column="embedding")
    .where("tenant_id = 42 AND status = 'published'")
    .nprobes(3)
    .limit(3)
    .select(["document_id", "title"])
)

hits = session.to_arrow(query)
print(hits)
```

Run it:

```bash
python quickstart.py
```

`documents.search(...)` creates an immutable query description. The selected
backend compiles it only when a terminal such as `to_arrow`, `collect`, or
`to_dataframe` is called. Results include the requested source columns and a
`_distance` column containing squared L2 distance; smaller values rank first.

## Inspect the Query

Use the same query value to inspect its logical and physical execution:

```python
print(session.explain(query))
print(session.analyze(query))
```

`explain` plans the query without running the final search. `analyze` executes
it and reports operator metrics.

## Use Your Own Parquet Table

Replace the packaged source with an absolute file URI, directory URI, or
wildcard pattern:

```python
session.register_parquet(
    "documents",
    "file:///data/documents/*/part-*.parquet",
)
```

Every vector value must be a non-null, fixed-dimension Parquet list whose
elements are non-null, finite `float32` or `float64` values. Each key field must
identify source rows using an exact supported scalar type. See the
[IVF index schema](../spec/ivf/index-schema.md) for the normative source
requirements.

## Next Steps

- Follow the [local backend guide](guides/local.md) for persistence, query
  composition, build controls, refresh, caching, and maintenance.
- Read [core concepts](concepts.md) before sharing an index across engines.
- Review [configuration](configuration.md) before using S3, HDFS, Iceberg,
  Spark, or StarRocks.
- Check [current limitations](limitations.md) before planning a production
  deployment.

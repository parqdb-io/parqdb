<div align="center">
  <img src="https://raw.githubusercontent.com/petrizhang/relify/main/assets/relify-header.svg" alt="Relify" width="760">
  <p>
    <strong>Lightweight vector index extension for the open lakehouse stack.</strong>
  </p>
  <p>
    <a href="https://pypi.org/project/relify/"><img alt="PyPI" src="https://img.shields.io/pypi/v/relify.svg"></a>
    <a href="https://github.com/petrizhang/relify/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/petrizhang/relify/actions/workflows/ci.yml/badge.svg?branch=main"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/pyproject.toml"><img alt="Python 3.11-3.14" src="https://img.shields.io/badge/python-3.11--3.14-blue.svg"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/Cargo.toml"><img alt="Rust 1.96" src="https://img.shields.io/badge/rust-1.96-orange.svg"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT%20AND%20Apache--2.0-green.svg"></a>
  </p>
  <p>
    <a href="#quick-start">Quick Start</a> |
    <a href="#why-relify">Why Relify</a> |
    <a href="#compute-engines">Compute Engines</a> |
    <a href="#benchmarks">Benchmarks</a> |
    <a href="#documentation">Documentation</a>
  </p>
</div>

---

Relify is an open-source Python and Rust library for indexing and searching
lakehouse data with the compute engines you already use. It stores vector
indexes as open Parquet or Iceberg tables, allowing DataFusion, StarRocks, and
Spark to query them directly with SQL while source data stays where it is.

Relify targets analytical and offline vector workloads such as large-k
retrieval, similarity joins, and vector search composed with analytical
queries. Dedicated vector databases remain the better fit for latency-sensitive,
high-concurrency online serving.

## Quick Start

Relify supports standard CPython 3.11 through 3.14 on Linux x86_64 and macOS
arm64. Install the embedded DataFusion and Parquet path:

```bash
python -m pip install relify
```

From a new working directory, build an IVF-Flat index over the dataset included
in the package and run a filtered vector query:

```python
import relify

session = relify.connect("./relify-data")
session.register_parquet("documents", relify.datasets.uri("documents"))
documents = session.table("documents")

documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=relify.IVF(nlist=3),
)
documents.wait_for_index("documents_embedding")

query = (
    documents.search([0.2, 0.0], column="embedding")
    .where("tenant_id = 42 AND status = 'published'")
    .nprobes(3)
    .limit(3)
    .select(["document_id", "title", "category"])
)

print(session.collect(query).to_pylist())
```

Vector search remains a relation rather than a terminal service call. Keep the
query lazy, register it as a DataFusion view, and continue with SQL in the same
execution context:

```python
session.register_parquet(
    "document_stats",
    relify.datasets.uri("document_stats"),
)
session.register_view("vector_hits", session.to_dataframe(query))

summary = session.sql("""
    SELECT
        h.category,
        COUNT(*) AS matches,
        AVG(h._distance) AS avg_distance,
        MAX(s.popularity) AS max_popularity
    FROM vector_hits AS h
    JOIN document_stats AS s USING (document_id)
    GROUP BY h.category
    ORDER BY h.category
""")
print(summary.to_pydict())
```

The packaged dataset makes this example self-contained. The
[getting-started guide](https://github.com/petrizhang/relify/blob/main/docs/getting-started.md)
covers persistent tables, existing indexes, query inspection, and source schema
requirements.

## Why Relify

- **Zero ETL into a vector database.** Source vectors stay in their existing
  lakehouse tables; Relify writes only index data and metadata.
- **One open vector index.** IVF centroids and postings are ordinary relational
  data, published as Parquet datasets or Iceberg tables rather than an
  engine-owned binary artifact.
- **Across compute engines.** Engine-specific backends consume the same index
  model and query contract instead of maintaining a separate copy per runtime.
- **SQL-native execution.** Cluster pruning, source filtering, joins, distance
  computation, and top-k remain inside the host engine's relational plan.

## Compute Engines

| Engine | Model | Current capability | Status |
| --- | --- | --- | --- |
| DataFusion | Embedded | Build and query Parquet indexes in one Python process | Supported |
| Spark Classic | Batch | Build Iceberg indexes; query Parquet and Iceberg | Experimental |
| StarRocks | OLAP | Query Spark-built Iceberg indexes over Arrow Flight SQL | Experimental |

DataFusion is the default backend. Spark and StarRocks live under
`relify.experimental` and require caller-managed engines and catalog
configuration. All three use the same query model and open index metadata.

See the [local](https://github.com/petrizhang/relify/blob/main/docs/guides/local.md),
[Spark](https://github.com/petrizhang/relify/blob/main/docs/guides/spark.md), and
[StarRocks](https://github.com/petrizhang/relify/blob/main/docs/guides/starrocks.md)
guides for installation and configuration.

## How It Works

![Relify builds an open index beside the source table and queries both with the host compute engine](https://raw.githubusercontent.com/petrizhang/relify/04c9ddc1c90ebee5626e66d96a4caa7fc9c55c4e/assets/how-it-works.svg)

Source rows remain in their original Parquet or Iceberg table. Building an
IVF-Flat index writes only portable metadata, centroids, and postings as open
table data.

At query time, an engine-specific adapter binds the source and index to
DataFusion, StarRocks, or Spark. The engine performs candidate pruning, source
filtering, distance calculation, top-k, and subsequent analytical SQL in its
own runtime.

The [open index specification](https://github.com/petrizhang/relify/blob/main/spec/README.md)
defines the shared schema and query semantics; the
[architecture guide](https://github.com/petrizhang/relify/blob/main/docs/architecture.md)
describes the implementation boundaries.

## Benchmarks

The current reproducible benchmark compares persisted, single-node IVF-Flat
construction and memory-resident large-k search on one million 128-dimensional
vectors. It ran on a 10-core Apple M4 with 16 GB of unified memory.

![Persisted IVF-Flat Build Time](https://raw.githubusercontent.com/petrizhang/relify/main/assets/build-time.svg)

![Large-k IVF Recall-Latency](https://raw.githubusercontent.com/petrizhang/relify/main/assets/search-recall-latency.svg)

Both implementations start from the same uncompressed Parquet source. Query
measurements use one query at a time, `nlist=4,096`, increasing `nprobe`, and
`k=10,000`, `20,000`, and `100,000`. See the
[methodology and raw results](https://github.com/petrizhang/relify/tree/main/benchmarks/results/macos-arm64-2026-07-29/)
for complete measurements.

## Documentation

- [Getting started](https://github.com/petrizhang/relify/blob/main/docs/getting-started.md)
  and [Python examples](https://github.com/petrizhang/relify/tree/main/examples/python)
- [Core concepts](https://github.com/petrizhang/relify/blob/main/docs/concepts.md),
  [architecture](https://github.com/petrizhang/relify/blob/main/docs/architecture.md),
  and [open index specification](https://github.com/petrizhang/relify/blob/main/spec/README.md)
- [Python API](https://github.com/petrizhang/relify/blob/main/docs/python-api.md)
  and [configuration](https://github.com/petrizhang/relify/blob/main/docs/configuration.md)
- [Current limitations](https://github.com/petrizhang/relify/blob/main/docs/limitations.md),
  [troubleshooting](https://github.com/petrizhang/relify/blob/main/docs/troubleshooting.md),
  and [roadmap](https://github.com/petrizhang/relify/blob/main/docs/roadmap.md)

## TEngineDB-V

Relify began as the open-source research prototype behind TEngineDB-V. The
project now develops those ideas into a general-purpose vector extension for
the open lakehouse stack.

## Development

Relify uses uv, Maturin, Cargo, and a small Makefile orchestration layer:

```bash
make sync
make develop
make check
```

See [CONTRIBUTING.md](https://github.com/petrizhang/relify/blob/main/CONTRIBUTING.md)
for quality gates, fixtures, benchmarks, and contribution guidelines.

## License

Relify's original code is available under the
[MIT License](https://github.com/petrizhang/relify/blob/main/LICENSE). Wheels
include the vendored DataFusion Python binding under Apache-2.0; see the
[third-party notices](https://github.com/petrizhang/relify/blob/main/THIRD_PARTY_NOTICES.md).

Relify builds on work from LanceDB, DataFusion, DuckDB, StarRocks, Apache Spark,
and Apache Iceberg, with gratitude to their contributors and communities.

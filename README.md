<div align="center">
  <img src="https://raw.githubusercontent.com/petrizhang/relify/main/assets/relify-header.svg" alt="Relify" width="760">
  <p>
    <strong>Lightweight vector index extension for the open lakehouse stack.</strong>
  </p>
  <p>
    <a href="https://github.com/petrizhang/relify/blob/main/pyproject.toml"><img alt="Python 3.11-3.14" src="https://img.shields.io/badge/python-3.11--3.14-blue.svg"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/Cargo.toml"><img alt="Rust 1.96" src="https://img.shields.io/badge/rust-1.96-orange.svg"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/spec/ivf/index-schema.md"><img alt="Open index format" src="https://img.shields.io/badge/index-open%20format-22c55e.svg"></a>
    <a href="https://github.com/petrizhang/relify/blob/main/spec/ivf/query.md"><img alt="SQL native" src="https://img.shields.io/badge/search-SQL--native-0ea5e9.svg"></a>
  </p>
  <p>
    <a href="#why-relify">Why Relify</a> |
    <a href="#quick-start">Quick Start</a> |
    <a href="#how-it-works">How It Works</a> |
    <a href="#benchmarks">Benchmarks</a> |
    <a href="#documentation">Documentation</a>
  </p>
</div>

---

Relify is an open-source library for vector indexing and search in the
lakehouse. It stores vector indexes in open table formats, so embedded, OLAP,
and batch engines can query the same index directly with SQL, without deploying
a dedicated vector database or maintaining complex ETL pipelines.

While dedicated vector databases are optimized for latency-sensitive,
high-concurrency online serving, Relify focuses on analytical and offline vector
workloads that fit naturally in lakehouse engines: large-k retrieval, similarity
joins, and vector search combined with complex analytical queries.

## Why Relify

Most vector systems bind the index to one runtime: a binary file, a serving
stack, or a database-specific extension. That makes vector search hard to share
across engines and hard for query optimizers to reason about.

Relify takes the opposite path:

- **Zero ETL.** Build indexes alongside lakehouse tables without copying source
  data into a separate vector database or maintaining another ingestion
  pipeline.
- **One open vector index.** The logical index is materialized as open
  relational data: Parquet datasets locally and Iceberg tables through Spark.
- **Multiple compute engines.** Build and maintain one index, then choose the
  right embedded, OLAP, or batch engine for each workload without rebuilding or
  copying the index.
- **SQL-native search.** ANN search is decomposed into scans, filters, joins,
  aggregations, distance estimation, and top-k, so host engines can reuse their
  storage, scheduler, optimizer, cache, and execution runtime.

## Use Relify Your Way

- **Start locally.** Run Relify as an embedded library on a laptop or a single
  machine with DataFusion—no service deployment or external cluster required.
- **Query with an OLAP engine.** Use the experimental StarRocks integration to
  combine vector search with analytical queries over existing Iceberg tables.
- **Build with distributed compute.** Use the experimental Spark integration
  for distributed index construction and native DataFrame queries over
  Parquet and Iceberg tables.

The index format stays the same. Choose the execution environment that matches
how you want to use it.

The embedded DataFusion path is the stable implementation. Experimental Spark
Classic and StarRocks integrations are bundled under `relify.experimental`.
Spark queries Parquet and Iceberg indexes and writes Iceberg indexes; StarRocks
queries Spark-built Iceberg indexes through Arrow Flight SQL.

## Quick Start

Relify 0.1 supports standard CPython 3.11 through 3.14 on Linux x86_64
(`manylinux_2_28`) and macOS arm64 11 or later. Install the local DataFusion
and Parquet path:

```bash
python -m pip install relify
```

Optional compute integrations are installed separately:

```bash
python -m pip install "relify[iceberg]"
python -m pip install "relify[spark]"
python -m pip install "relify[starrocks]"
```

The extras install client libraries; they do not deploy Spark or StarRocks or
configure an Iceberg catalog.

### Vector Search

```python
import relify

session = relify.connect("./relify-data")
session.register_parquet(
    "documents",
    relify.datasets.uri("documents"),
)
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
    .select(["document_id", "title"])
)

hits = session.collect(query)
print(hits)
```

The synthetic `documents` table ships with Relify, so this path runs without
downloading or preparing data. The snippet assumes a new `./relify-data`
directory; the complete getting-started guide covers reopening existing tables
and indexes.

### Continue with DataFusion

When the results need further analysis, leave `query` uncollected and compile
it into a lazy DataFusion DataFrame with `session.to_dataframe(query)`. Vector
routing, source filtering, joins, and aggregation then remain in one logical
plan instead of materializing an intermediate hits table. DataFusion can
optimize the complete plan with projection and predicate pushdown, legal join
reordering, repartitioning, and runtime filters supported by the chosen join
strategy.

```python
from relify.datafusion import col, functions

hits_df = session.to_dataframe(query)
document_stats = session.read_parquet(relify.datasets.uri("document_stats"))

result = (
    hits_df.join(document_stats, on="document_id")
    .aggregate(
        "category",
        [
            functions.count(col("document_id")).alias("matches"),
            functions.avg(col("_distance")).alias("avg_distance"),
            functions.max(col("popularity")).alias("max_popularity"),
        ],
    )
    .sort("category")
    .collect()
)
```

The `document_stats` table is the second small dataset included in the package.

See the [Python API](https://github.com/petrizhang/relify/blob/main/docs/python-api.md)
for asynchronous builds, refresh,
exact-search fallback, query composition, catalog recovery, and maintenance.
The complete [getting-started guide](https://github.com/petrizhang/relify/blob/main/docs/getting-started.md)
explains written state, source requirements, query inspection, and next steps.

## Python Examples

Runnable examples are grouped by backend:

- [Local DataFusion](https://github.com/petrizhang/relify/tree/main/examples/python/local)
  covers the quick start, Parquet
  persistence, exact search, query plans, analytical composition, and index
  lifecycle.
- [Experimental Spark](https://github.com/petrizhang/relify/tree/main/examples/python/spark)
  demonstrates Iceberg index
  construction and native PySpark queries.
- [Experimental StarRocks](https://github.com/petrizhang/relify/tree/main/examples/python/starrocks)
  demonstrates query-only
  access to the same Spark-built Iceberg index over Arrow Flight SQL.

See the [examples guide](https://github.com/petrizhang/relify/blob/main/examples/python/README.md)
for prerequisites and commands.

## Documentation

Start from the [documentation index](https://github.com/petrizhang/relify/blob/main/docs/README.md),
which separates runnable workflows from API reference and project internals:

- [Getting started](https://github.com/petrizhang/relify/blob/main/docs/getting-started.md)
- [Core concepts](https://github.com/petrizhang/relify/blob/main/docs/concepts.md)
- [Local DataFusion and Parquet](https://github.com/petrizhang/relify/blob/main/docs/guides/local.md)
- [Experimental Spark and Iceberg](https://github.com/petrizhang/relify/blob/main/docs/guides/spark.md)
- [Experimental StarRocks and Iceberg](https://github.com/petrizhang/relify/blob/main/docs/guides/starrocks.md)
- [Configuration](https://github.com/petrizhang/relify/blob/main/docs/configuration.md)
- [Troubleshooting](https://github.com/petrizhang/relify/blob/main/docs/troubleshooting.md)
- [Current limitations](https://github.com/petrizhang/relify/blob/main/docs/limitations.md)
- [Python API](https://github.com/petrizhang/relify/blob/main/docs/python-api.md)
- [Open index specification](https://github.com/petrizhang/relify/blob/main/spec/README.md)

## How It Works

![Relational execution of an IVF vector query](https://raw.githubusercontent.com/petrizhang/relify/main/assets/relational-ivf-query.jpg)

*Figure from TEngineDB-V, illustrating an IVF-PQ/FastScan query as relational
stages: prune clusters, prepare distance lookups, join candidates, estimate
distances, and produce Top-K results. Relify currently implements IVF-Flat;
the PQ, lookup-table, and FastScan stages shown here are not yet implemented.*

Relify represents a vector index as open relational data instead of an opaque
artifact owned by one database. The same index can be built once, inspected
directly, queried from different engines, and composed with ordinary SQL
filters, joins, and aggregations.

The current implementation uses Parquet for index data, SQLite for the catalog,
and DataFusion for query execution. See the
[architecture](https://github.com/petrizhang/relify/blob/main/docs/architecture.md)
for component boundaries.

## Benchmarks

This benchmark measures embedded, single-node IVF-Flat build time and
memory-resident query latency. It ran on a MacBook Air with a 10-core Apple M4
CPU (4 performance and 6 efficiency cores) and 16 GB of unified memory.

![Persisted IVF-Flat Build Time](https://raw.githubusercontent.com/petrizhang/relify/main/assets/build-time.svg)

![Large-k IVF Recall-Latency](https://raw.githubusercontent.com/petrizhang/relify/main/assets/search-recall-latency.svg)

- **Faiss:** `IndexIVFFlat` uses 10 OpenMP threads with `parallel_mode=1`,
  parallelizing a single query across inverted lists.
- **Relify:** queries run through the embedded DataFusion session.
- **Memory residency:** complete persisted indexes are resident as decoded
  Arrow buffers in Relify and an in-process index in Faiss.

The build timer starts from the same uncompressed Parquet source and stops
after index persistence. The query curve measures one query at a time while
increasing `nprobe` at `k=10,000`, `20,000`, and `100,000`, with `nlist=4,096`.
See the [raw results](https://github.com/petrizhang/relify/tree/main/benchmarks/results/macos-arm64-2026-07-29/)
for the exact methodology and measurements.

## TEngineDB-V

Relify began as an open-source research prototype inspired by TEngineDB-V. The
project now develops those ideas into a general-purpose vector extension for
the open lakehouse stack.

## Status

| Goal | Guide | Status |
| --- | --- | --- |
| Build and query Parquet indexes in one Python process | [Local DataFusion and Parquet](https://github.com/petrizhang/relify/blob/main/docs/guides/local.md) | Stable |
| Build and query Iceberg indexes with Spark Classic | [Spark and Iceberg](https://github.com/petrizhang/relify/blob/main/docs/guides/spark.md) | Experimental |
| Query a Spark-built Iceberg index with StarRocks | [StarRocks and Iceberg](https://github.com/petrizhang/relify/blob/main/docs/guides/starrocks.md) | Experimental |
| Run the maintained examples | [Python examples](https://github.com/petrizhang/relify/blob/main/examples/python/README.md) | Tested in the repository |

The local DataFusion backend is the default place to begin. Spark and
StarRocks use the same query model and index metadata but require external
engine and catalog configuration.

The next milestone adds remote catalogs and production Spark coordination. See
the [current limitations](https://github.com/petrizhang/relify/blob/main/docs/limitations.md)
and [roadmap](https://github.com/petrizhang/relify/blob/main/docs/roadmap.md) for
scope and sequencing.

## Development

Python environments and mixed Python/Rust builds use
[uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/). Rust compilation remains managed by Cargo.
The Makefile only orchestrates those tools.

```bash
# Resolve and install the locked Python dependencies.
make sync

# Build and install the Rust-backed Python package in the current environment.
make develop

# Run formatting, linting, type checks, and the Rust and Python test suites.
make check
```

See [CONTRIBUTING.md](https://github.com/petrizhang/relify/blob/main/CONTRIBUTING.md)
for quality gates, fixtures, benchmarks, and contribution guidelines.

## Acknowledgements

Relify has learned a great deal from LanceDB, DataFusion, DuckDB, StarRocks,
Apache Spark, and Apache Iceberg, and we are deeply grateful to their
contributors and communities.

## License

Relify's original code is licensed under the
[MIT License](https://github.com/petrizhang/relify/blob/main/LICENSE). Python
wheels include the vendored DataFusion Python binding under Apache-2.0; see the
[third-party notices](https://github.com/petrizhang/relify/blob/main/THIRD_PARTY_NOTICES.md).

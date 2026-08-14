<div align="center">
  <img src="https://raw.githubusercontent.com/petrizhang/relify/main/assets/relify-header.svg" alt="Relify" width="760">
  <p>
    English |
    <a href="https://github.com/petrizhang/relify/blob/main/README.zh-CN.md">中文</a>
  </p>
  <p>
    <strong>Open vector indexes for the SQL engines you already use.</strong>
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
    <a href="#documentation">Documentation</a>
  </p>
</div>

---

Relify is an open-source Python and Rust library for indexing and searching
lakehouse data with the compute engines you already use. It stores vector
indexes as open Parquet or Iceberg tables, allowing DataFusion, StarRocks, and
Spark to query them directly with SQL while source data stays where it is.

If Relify is useful to you, a ⭐ helps others find the project.

## Quick Start

Relify supports standard CPython 3.11 through 3.14 on Linux x86_64 and macOS
arm64. Install the embedded DataFusion and Parquet path:

```bash
python -m pip install relify
```

Spark and StarRocks are optional integrations with separate setup; see
[Compute Engines](#compute-engines).

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

## TEngineDB-V and Relify

[TEngineDB-V: An OLAP-Native Vector Search System for Large-k Workloads at
Tencent](https://arxiv.org/abs/2608.00650) is Tencent's production system for
large-k vector search. On a 10-billion-vector deployment, its deep integration
with TEngineDB delivers up to a 52x speedup over the legacy system.

<p align="center">
  <img src="https://raw.githubusercontent.com/petrizhang/relify/main/assets/tenginedb-v-figure-7.png" alt="Figure 7: Latency-Recall Trade-off Across Systems" width="760">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/petrizhang/relify/main/assets/tenginedb-v-figure-13.png" alt="Figure 13: Production performance at 10-billion scale" width="760">
</p>

Relify shares the idea, not the implementation. It rebuilds table-native vector
search around open index formats and existing SQL engines, aiming for
TEngineDB-V-class performance without requiring a proprietary engine.

If you use Relify in your research, please cite our VLDB 2026 Industry Track
paper:

```bibtex
@misc{wu2026tenginedbvolapnativevectorsearch,
  title         = {{TEngineDB-V}: An {OLAP}-Native Vector Search System for Large-$k$ Workloads at Tencent},
  author        = {Xufei Wu and Pengcheng Zhang and Yitong Song and Xiaobo Zhang and Anqi Liang and Kai Wang and Jijun Du and Yidi Xiong and Guangxu Cheng and Zhe Chen and Peng Chen and Guoliang Li and Xuanhe Zhou and Fan Wu},
  year          = {2026},
  eprint        = {2608.00650},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  url           = {https://arxiv.org/abs/2608.00650},
}
```

## Development

Relify's next phase is being designed in public. We welcome concrete use cases,
benchmark results, design feedback, and implementation help:

- [Narrow the product around an embedded vector lakehouse](https://github.com/petrizhang/relify/issues/9)
- [Improve storage-backed Parquet search](https://github.com/petrizhang/relify/issues/8)
  and [measure the online serving envelope](https://github.com/petrizhang/relify/issues/11)
- [Add an extensible index-family framework](https://github.com/petrizhang/relify/issues/10)
- [Design compute-storage separation](https://github.com/petrizhang/relify/issues/13)
  and [build a complete DuckLake workflow](https://github.com/petrizhang/relify/issues/12)

If you are working on RAG, agent trajectory storage, Parquet performance, or
embedded lakehouse systems, share your workload and requirements in the
relevant issue. Comment before starting a large change so that scope and
interfaces can be agreed on first.

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

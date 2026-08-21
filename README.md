<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/parqdb-io/parqdb/main/assets/parqdb/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/parqdb-io/parqdb/main/assets/parqdb/logo.svg" alt="ParqDB" width="520">
  </picture>
  <p>
    English |
    <a href="https://github.com/parqdb-io/parqdb/blob/main/README.zh-CN.md">中文</a>
  </p>
  <p>
    <strong>Billion-scale embedded vector database built entirely on Parquet and Arrow.</strong>
  </p>
  <p>
    <a href="https://pypi.org/project/parqdb/"><img alt="PyPI" src="https://img.shields.io/pypi/v/parqdb.svg"></a>
    <a href="https://github.com/parqdb-io/parqdb/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/parqdb-io/parqdb/actions/workflows/ci.yml/badge.svg?branch=main"></a>
    <a href="https://github.com/parqdb-io/parqdb/blob/main/pyproject.toml"><img alt="Python 3.11-3.14" src="https://img.shields.io/badge/python-3.11--3.14-blue.svg"></a>
    <a href="https://github.com/parqdb-io/parqdb/blob/main/Cargo.toml"><img alt="Rust 1.96" src="https://img.shields.io/badge/rust-1.96-orange.svg"></a>
    <a href="https://github.com/parqdb-io/parqdb/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT%20AND%20Apache--2.0-green.svg"></a>
  </p>
  <p>
    <a href="https://search.parqdb.io/">Browser Demo</a> |
    <a href="#quick-start">Quick Start</a> |
    <a href="#status">Status</a> |
    <a href="#documentation">Documentation</a>
  </p>
</div>

---

ParqDB is an embedded vector database for larger-than-memory search and analytics
on billion-scale multimodal data, with Parquet storage and Arrow-native execution.

<p align="center">
  <a href="https://search.parqdb.io/">
    <img src="assets/browser-demo-v2.gif" alt="ParqDB querying a published Wikipedia vector index directly from the browser" width="960">
  </a>
  <br>
  <a href="https://search.parqdb.io/"><strong>Try the live browser demo →</strong></a>
  <br>
  <sub>IVF-LVQ8 over HTTP Range · Parquet · WebAssembly · no query server</sub>
</p>

<p align="center">
  <sub>⭐ If ParqDB is useful, star the repo to help more people find it.</sub>
</p>

**Key Features**

- **Billion-scale search in bounded memory.** Search one billion vectors
  ([SIFT1B](benchmarks/results/linux-x86_64-2026-08-17/README.md)) at 90.3%
  recall with 63.05 ms median latency using just 2 CPU cores and 4 GB of memory.
- **Everything is Parquet.** Source data and vector indexes use standard Parquet
  rather than proprietary binary formats, making indexes easy to version,
  publish, and share across engines and applications.
- **Publish once, query anywhere.** Publish immutable IVF-LVQ indexes to object
  storage and search them directly from a browser over HTTP Range and
  WebAssembly, without a query server.
- **Multimodal data, SQL-native search.** Vector search is expressed as relational
  operations, allowing the SQL optimizer to combine it with filters, joins, and
  aggregations in a single execution plan.
- **Built for serving and analytics.** Use intra-query parallelism for
  low-latency analytical and large-k searches, and inter-query parallelism for
  high-throughput online serving.
- **Scale from one core to thousands.** Run embedded on a single machine, then
  use the same Parquet index with Spark or StarRocks at cluster scale.

## Quick Start

Install ParqDB:

```bash
python -m pip install parqdb
```

From a new working directory, build a source-encoded IVF index over the dataset
included in the package and run a filtered vector query:

```python
import parqdb

session = parqdb.connect("./parqdb-data")
session.register_parquet("documents", parqdb.datasets.uri("documents"))
documents = session.table("documents")

documents.create_index(
    "documents_embedding",
    column="embedding",
    key=["document_id"],
    config=parqdb.IVF(nlist=3),
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

Vector search remains relational rather than becoming a terminal service call.
Compile it as a SQL subquery and compose it with the rest of the analysis:

```python
session.register_parquet(
    "document_stats",
    parqdb.datasets.uri("document_stats"),
)
search_sql = session.to_sql(query)
summary = session.sql(f"""
    SELECT
        h.category,
        COUNT(*) AS matches,
        AVG(h._distance) AS avg_distance,
        MAX(s.popularity) AS max_popularity
    FROM ({search_sql}) AS h
    JOIN document_stats AS s USING (document_id)
    GROUP BY h.category
    ORDER BY h.category
""")
print(summary.to_pydict())
```

The packaged dataset makes this example self-contained. The
[getting-started guide](https://github.com/parqdb-io/parqdb/blob/main/docs/getting-started.md)
covers persistent tables, existing indexes, query inspection, and source schema
requirements.

Publish a source table and an immutable browser index in one command. For raw
text, ParqDB uses the pinned MiniLM ONNX model for both offline embeddings and
browser parity metadata, then builds hierarchical IVF-LVQ8 and uploads every
object before exposing `index/manifest.json`:

```bash
python -m pip install "parqdb[publish]"

parqdb publish \
  --source documents.parquet \
  --key chunk_id \
  --text-column title \
  --text-column section \
  --text-column text \
  --nlist 4096 \
  --destination s3://my-bucket/kb/v1 \
  --s3-endpoint https://ACCOUNT_ID.r2.cloudflarestorage.com \
  --s3-region auto \
  --public-url https://data.example.com/kb/v1
```

Credentials come from the standard `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` environment variables. If `documents.parquet` already
contains embeddings, replace the three `--text-column` options with
`--vector-column embedding`. Publication refuses to overwrite an existing
prefix and verifies public HTTP Range and CORS behavior before succeeding.

For a complete document-to-GitHub-Pages knowledge base, including token-aware
chunking and the search UI, see
[`parqdb-knowledgebase`](https://github.com/parqdb-io/parqdb-knowledgebase).

## Status

| Runtime | Storage | Current capability | Status |
| --- | --- | --- | --- |
| Embedded DataFusion | Parquet | Build and query IVF, IVF-LVQ4, and IVF-LVQ8 indexes | Supported |
| Browser/WASM | Public HTTPS object storage | Query immutable IVF-LVQ4 and IVF-LVQ8 indexes over HTTP Range | Experimental |
| Embedded DataFusion | Iceberg | Query exact table snapshots through PyIceberg | Experimental |
| Client/server | Authorized Parquet sources | Build and query through the HTTP API | Experimental |

The first supported product surface is the embedded DataFusion runtime. The
index specification remains independent of that runtime; distributed engine
adapters are no longer bundled into the Python package.

See the [local guide](https://github.com/parqdb-io/parqdb/blob/main/docs/guides/local.md)
for installation and configuration.

The experimental HTTP server is documented in the
[server guide](https://github.com/parqdb-io/parqdb/blob/main/docs/guides/server.md).

## Documentation

- [Getting started](https://github.com/parqdb-io/parqdb/blob/main/docs/getting-started.md)
  and [Python examples](https://github.com/parqdb-io/parqdb/tree/main/examples/python)
- [Core concepts](https://github.com/parqdb-io/parqdb/blob/main/docs/concepts.md),
  [architecture](https://github.com/parqdb-io/parqdb/blob/main/docs/architecture.md),
  and [open index specification](https://github.com/parqdb-io/parqdb/blob/main/spec/README.md)
- [Python API](https://github.com/parqdb-io/parqdb/blob/main/docs/python-api.md)
  and [configuration](https://github.com/parqdb-io/parqdb/blob/main/docs/configuration.md),
  including the [server guide](https://github.com/parqdb-io/parqdb/blob/main/docs/guides/server.md)
- [Current limitations](https://github.com/parqdb-io/parqdb/blob/main/docs/limitations.md),
  [troubleshooting](https://github.com/parqdb-io/parqdb/blob/main/docs/troubleshooting.md),
  and [roadmap](https://github.com/parqdb-io/parqdb/blob/main/docs/roadmap.md)

## TEngineDB-V and ParqDB

[TEngineDB-V: An OLAP-Native Vector Search System for Large-k Workloads at
Tencent](https://arxiv.org/abs/2608.00650) is Tencent's production system for
large-k vector search. On a 10-billion-vector deployment, its deep integration
with TEngineDB delivers up to a 52x speedup over the legacy system.

<p align="center">
  <img src="https://raw.githubusercontent.com/parqdb-io/parqdb/main/assets/tenginedb-v-figure-7.png" alt="Figure 7: Latency-Recall Trade-off Across Systems" width="760">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/parqdb-io/parqdb/main/assets/tenginedb-v-figure-13.png" alt="Figure 13: Production performance at 10-billion scale" width="760">
</p>

ParqDB shares the idea, not the implementation. It rebuilds table-native vector
search around open index formats and existing SQL engines, aiming for
TEngineDB-V-class performance without requiring a proprietary engine.

If you use ParqDB in your research, please cite our VLDB 2026 Industry Track
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

ParqDB's next phase is being designed in public. We welcome concrete use cases,
benchmark results, design feedback, and implementation help:

- [Narrow the product around an embedded vector lakehouse](https://github.com/parqdb-io/parqdb/issues/9)
- [Improve storage-backed Parquet search](https://github.com/parqdb-io/parqdb/issues/8)
  and [measure the online serving envelope](https://github.com/parqdb-io/parqdb/issues/11)
- [Add an extensible index-family framework](https://github.com/parqdb-io/parqdb/issues/10)
- [Design compute-storage separation](https://github.com/parqdb-io/parqdb/issues/13)
  and [build a complete DuckLake workflow](https://github.com/parqdb-io/parqdb/issues/12)

If you are working on RAG, agent trajectory storage, Parquet performance, or
embedded lakehouse systems, share your workload and requirements in the
relevant issue. Comment before starting a large change so that scope and
interfaces can be agreed on first.

ParqDB uses uv, Maturin, Cargo, and a small Makefile orchestration layer:

```bash
make sync
make develop
make check
```

See [CONTRIBUTING.md](https://github.com/parqdb-io/parqdb/blob/main/CONTRIBUTING.md)
for quality gates, fixtures, benchmarks, and contribution guidelines.

## License

ParqDB's original code is available under the
[MIT License](https://github.com/parqdb-io/parqdb/blob/main/LICENSE). Wheels
include the vendored DataFusion Python binding under Apache-2.0; see the
[third-party notices](https://github.com/parqdb-io/parqdb/blob/main/THIRD_PARTY_NOTICES.md).

ParqDB builds on work from LanceDB, DataFusion, DuckDB, StarRocks, Apache Spark,
and Apache Iceberg, with gratitude to their contributors and communities.

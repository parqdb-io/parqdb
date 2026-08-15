# Roadmap

Relify develops one narrow, interoperable path at a time. The specification
defines portable behavior; the library implements a supported subset of it.

## 0.1.0: Initial Release

### Local DataFusion and Parquet

The initial release includes:

- SQLite catalog for persistent Parquet definitions and index mappings;
- exact paths and nested `*` patterns over `file`, S3, and HDFS sources;
- `file`, S3, and HDFS warehouses;
- Parquet source and index tables;
- local parallel IVF construction with shared centroids and source, LVQ4, or
  LVQ8 postings;
- squared-L2 and cosine search over `float32` and `float64` source vectors;
- DataFusion search, filtering, projection, and exact fallback;
- exact Iceberg snapshot reads for published indexes;
- independent DuckDB execution of the portable IVF fixtures;
- reproducible persisted-build and large-k Recall-latency results with a Faiss
  baseline;
- verified wheels with locked dependency SBOMs and installed-package smoke
  tests;
- immutable metadata publication and atomic refresh; and
- reachability-based orphan removal.

The release gates and procedure are defined in
[`CONTRIBUTING.md`](../CONTRIBUTING.md), [`SECURITY.md`](../SECURITY.md), and
[`release.md`](release.md).

### Spark and Iceberg

The initial release includes:

- a concrete Spark Classic session over a caller-owned `SparkSession`;
- exact Iceberg source and index snapshot resolution through PyIceberg;
- native PySpark DataFrame query plans with relational cluster routing;
- direct Spark reads of locally built Parquet indexes;
- shared index discovery, selection, and metadata loading through
  `relify-index`.

The bundled Spark integration is query-only. Before this path is
production-ready it still requires a maintained Spark/Iceberg conformance
environment, reproducible distributed benchmarks, and a remote Relify index
catalog.

### StarRocks Query

The query-only StarRocks integration includes:

- a concrete query-only session over a caller-owned Arrow Flight SQL ADBC
  connection;
- exact Iceberg table UUID, schema, and snapshot validation through PyIceberg;
- StarRocks SQL compilation with relational centroid routing, postings
  pruning, transparent source resolution, filtering, projection, and Top-K;
- Arrow-table collection and native `EXPLAIN`; and
- an opt-in StarRocks/Iceberg execution of the source-encoded L2 IVF fixture.

This path requires StarRocks 3.5.1 or later and one Iceberg catalog registered
under the same logical name in StarRocks and PyIceberg. It is intentionally
query-only and Iceberg-only at the StarRocks execution layer; construction may
be delegated to a compatible third-party builder. Before production use it still
requires a maintained conformance deployment, reproducible StarRocks
benchmarks, and a remote Relify index catalog.

### Backend Extension API

The initial release makes additional query engines independently integrable
without changing the Relify package:

- lazy discovery through the `relify.backends` package entry-point group;
- concrete, caller-owned session factories rather than a universal backend
  container;
- typed static and bound-session capability reports;
- shared immutable IVF query resolution through `ResolvedSearch`;
- a portable `pyarrow.Table` collection terminal whose empty results preserve
  schema; and
- reusable backend query-contract checks outside the specification.

The stable local and bundled experimental Spark and StarRocks sessions publish
capabilities through the same API.

Index construction is independently extensible through
`relify.builders.v1`. The bundled local builder and compatible third-party
builders publish typed source/output profiles, while every concrete table
shares the same asynchronous lifecycle.

## Next: Remote Catalogs and Spark Connect

The next milestone adds a remotely coordinated Relify index catalog and
defines a separate Spark Connect operation lifecycle. New host-engine
integrations must consume the shared fixtures and produce the same ordered
query results as the built-in execution paths.

## Later

Additional index families, distance metrics, SDK languages, construction using
StarRocks compute, Parquet queries through StarRocks, and managed services
remain out of scope until the implemented paths have independent conformance
tests and reproducible benchmarks.

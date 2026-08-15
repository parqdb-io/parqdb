# Roadmap

Relify develops one narrow, interoperable path at a time. The specification
defines portable behavior; the library implements the supported subset below.

## Current: Shared IVF

The current local DataFusion path provides:

- persistent Parquet sources and indexes over `file`, S3, and HDFS storage;
- one shared centroid artifact per source state, vector field,
  distance metric, and `nlist`;
- source, LVQ4, and LVQ8 logical indexes that reuse the shared IVF artifact;
- squared-L2 and cosine search over `list<float>` and `list<double>` source
  vectors;
- filtering, projection, exact fallback, SQL and DataFrame composition;
- immutable metadata publication, atomic refresh, catalog recovery, and
  retention-aware orphan removal; and
- open conformance fixtures exercised independently through DuckDB.

The experimental Spark and StarRocks integrations are query-only. They consume
compatible source-encoded, squared-L2 indexes through native engine plans.
Spark reads Parquet and Iceberg relations; StarRocks reads Iceberg relations.

## Next: Validate the Storage Path

The next milestone focuses on evidence and reliability before expanding the
surface area:

- benchmark source, LVQ4, and LVQ8 search under constrained memory;
- validate shared-IVF reuse and cosine recall on public datasets;
- complete the decompressed Parquet page-cache design and implementation;
- define a conforming Iceberg builder for the shared-IVF schema; and
- establish maintained Spark and StarRocks conformance environments.

## Later

Batch queries, additional index families, remote catalogs, Spark Connect,
construction with distributed engines, and additional SDK languages remain
out of scope until the current paths have reproducible benchmarks and stable
conformance tests.

# Relify Documentation

Relify is a Python and Rust library for building open-format vector indexes and
querying them with DataFusion, Spark, and StarRocks. The documentation starts
with runnable workflows, then separates operational reference material from
the index specification and implementation design.

## Start Here

- [Getting started](getting-started.md) installs Relify, builds a local IVF
  index, and runs a filtered vector query.
- [Core concepts](concepts.md) explains source tables, open indexes, catalogs,
  snapshots, builders, and compute backends.
- [Current limitations](limitations.md) defines the supported 0.1 boundary
  before you choose an architecture.

## Choose a Workflow

| Goal | Guide | Status |
| --- | --- | --- |
| Build and query Parquet indexes in one Python process | [Local DataFusion and Parquet](guides/local.md) | Stable |
| Build and query Iceberg indexes with Spark Classic | [Spark and Iceberg](guides/spark.md) | Experimental |
| Query a Spark-built Iceberg index with StarRocks | [StarRocks and Iceberg](guides/starrocks.md) | Experimental |
| Run the maintained examples | [Python examples](../examples/python/README.md) | Tested in the repository |

The local backend is the default place to begin. Spark and StarRocks use the
same query model and index metadata but require external engine and catalog
configuration.

## Reference

- [Python API](python-api.md): sessions, tables, index lifecycle, search,
  DataFusion composition, catalogs, and maintenance.
- [Configuration](configuration.md): Python versions, optional dependencies,
  catalogs, storage, builders, and backend-specific settings.
- [Troubleshooting](troubleshooting.md): installation, storage, catalog,
  indexing, query, Spark, and StarRocks failures.
- [Backend extension API](backends.md): build a third-party compute-engine
  adapter against Relify's versioned backend contract.
- [Open index specification](../spec/README.md): the portable metadata,
  relation schemas, storage profiles, and query semantics.

## Project Internals

- [Architecture](architecture.md)
- [RFCs](rfcs/)
- [Roadmap](roadmap.md)
- [Architecture decisions](decisions/)
- [Contributing](../CONTRIBUTING.md)
- [Changelog](../CHANGELOG.md)
- [Release process](release.md)
- [Security policy](../SECURITY.md)

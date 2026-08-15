# Relify Documentation

Relify is a Python and Rust library for building open vector indexes over
lakehouse tables and querying them with an embedded SQL runtime.

## Start Here

- [Getting started](getting-started.md) installs Relify, builds an IVF index,
  and runs a filtered vector query.
- [Core concepts](concepts.md) explains source tables, open indexes, catalogs,
  snapshots, and query execution.
- [Current limitations](limitations.md) defines the supported boundary.

## Workflows

| Goal | Guide | Status |
| --- | --- | --- |
| Build and query Parquet indexes in one Python process | [Embedded DataFusion and Parquet](guides/local.md) | Supported |
| Query an existing catalog through HTTP | [Python API](python-api.md#query-an-existing-remote-catalog) | Experimental |
| Resolve exact Iceberg snapshots through PyIceberg | [Embedded DataFusion and Parquet](guides/local.md) | Experimental |
| Run maintained examples | [Python examples](../examples/python/README.md) | Tested |

## Reference

- [Python API](python-api.md)
- [Configuration](configuration.md)
- [Troubleshooting](troubleshooting.md)
- [Open index specification](../spec/README.md)

## Project Internals

- [Architecture](architecture.md)
- [RFCs](rfcs/)
- [Roadmap](roadmap.md)
- [Architecture decisions](decisions/)
- [Contributing](../CONTRIBUTING.md)
- [Release process](release.md)

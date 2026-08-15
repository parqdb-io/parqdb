# Independent Index Builders

- Status: Superseded by
  [`20260815-unified-embedded-client-server-api.md`](../rfcs/20260815-unified-embedded-client-server-api.md)
- Date: 2026-07-31

## Context

Relify originally attached construction to concrete query sessions. Local
tables accepted only `Local`, Spark tables constructed only through their own
Spark session, and StarRocks tables exposed no construction API. This made a
query engine appear to own index construction even though builders produce
portable index relations and query backends only consume them.

The intended composition is broader: a table queried through StarRocks may be
indexed by a caller-owned Spark session, and the resulting Iceberg index must
remain queryable by Spark, StarRocks, or DataFusion.

## Decision

Query backends and index builders are independent extension boundaries.
`relify.backends.v1` reports query, terminal, and maintenance capabilities.
`relify.builders.v1` defines builder identity, build profiles, pinned build
requests, portable build output, and the builder protocol.

Every concrete table reuses one table-centered lifecycle:
`create_index`, `index_status`, `wait_for_index`, `list_indexes`,
`drop_index`, and `search`. A session-owned `BuildCoordinator` resolves and
pins the source relation at submission, validates the selected builder
profile, owns asynchronous state, publishes returned artifacts through the
Relify index repository, and discards unpublished output after publication
failure.

Builder objects own construction compute and physical writes:

- `Local` builds Parquet indexes through the embedded Rust runtime.
- compatible third-party builders may build Iceberg indexes through their own
  execution runtime.

A local session defaults to `Local()`. The bundled experimental Spark and
StarRocks sessions currently have no default builder. Callers may pass a
compatible third-party builder explicitly.

`WriteOptions` contains output properties meaningful across builders:
partition count, Parquet compression, and target file size. Local-only thread,
row-group-row, and write-batch controls belong to `Local`.

The local Rust path retains its existing atomic build-and-publish operation
internally because its build lease and garbage-collection coordination share
one native transaction boundary. This is an implementation specialization
behind the common coordinator and does not couple the public builder to the
query backend.

## Consequences

Adding a builder no longer requires adding or modifying a query backend.
Adding a query backend no longer requires construction stubs or a build
capability matrix. Builder compatibility is checked against the pinned source
profile and produced index-table profile.

The coordinator is the single owner of Python-visible operation state and
catalog publication. Builder implementations must return portable relation
references and a cleanup action for unpublished data. A future remote build
service can implement the same protocol without changing table methods.

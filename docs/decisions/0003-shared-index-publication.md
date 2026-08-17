# Shared Index Publication

- Status: Superseded by [ADR 0004](0004-spark-iceberg-session.md)
- Date: 2026-07-30

## Context

Index builders produce immutable index relations, but publication has the same
portable state transition for every execution backend: construct a validated
metadata document, write it to managed storage, and make it current with a
catalog register or compare-and-swap commit.

Keeping that transition inside the embedded DataFusion implementation would
force another builder, such as Spark, to duplicate metadata sequencing,
warehouse layout, and catalog concurrency behavior. It also made source
metadata implicitly Parquet-specific.

## Decision

`parqdb-publish` owns immutable metadata storage and catalog publication. It
accepts a complete `RelationReference`, backend identity, and backend-produced
`IndexArtifacts`. It has no dependency on DataFusion, Arrow, Parquet, or a
construction engine.

Execution backends remain responsible for resolving and scanning source tables,
building index relations, and choosing their physical execution model. They
call the shared publisher only after all immutable index relations are
complete.

## Consequences

Local and future Spark builders publish identical metadata and use the same
catalog compare-and-swap behavior without sharing execution code. Iceberg and
Parquet sources use the same publication path because the publisher preserves
their portable relation reference instead of reducing it to a URI.

Failed publication may leave unreachable immutable objects. It cannot expose a
partial snapshot because the catalog update remains the publication point.

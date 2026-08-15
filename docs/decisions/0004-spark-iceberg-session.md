# Spark Iceberg Session

- Status: Superseded by [Shared IVF and Cosine Support](../rfcs/20260815-shared-ivf-cosine.md)
- Date: 2026-07-30

## Context

Relify needs a distributed builder and query implementation without turning
the embedded DataFusion session into a replaceable backend container. Spark
already owns table resolution, scheduling, relational optimization, and
Iceberg I/O. Relify still needs exact Iceberg table identity and snapshot
metadata that are not part of PySpark's public DataFrame API.

Index loading and selection were previously implemented inside
`relify-local`, while only metadata publication was shared. A Spark
implementation would therefore have duplicated catalog discovery and implicit
index-selection rules.

## Decision

`relify-index` owns backend-neutral index loading, discovery, selection,
immutable metadata storage, and catalog publication. It supersedes the narrower
`relify-publish` boundary from
[ADR 0003](0003-shared-index-publication.md).

Spark is a second concrete session under `relify.experimental.spark`; it is not a backend
object installed into the local session. A Spark session binds:

- one caller-owned Spark Classic `SparkSession`;
- one PyIceberg catalog;
- the same logical catalog name already configured in Spark; and
- one Relify index catalog.

`session.table("namespace.table")` verifies that both Spark and PyIceberg can
resolve the table. PyIceberg supplies the exact table UUID and snapshot ID.
Spark reads that snapshot and owns every source scan, shuffle, write, and query
plan. Relify does not instantiate DataFusion in a Spark session and does not
use private PySpark JVM handles.

Spark construction uses MLlib's block KMeans solver for distributed training.
Training uses a seeded distributed sample whose expected size is at most 256
points per centroid from the pinned snapshot, with a warning below 39 points
per centroid. Sampling preserves Iceberg snapshot and delete semantics without
collecting the source table on the driver. KMeans initialization uses a fixed
seed.
Relify does not cache the source table around MLlib's own persisted training
blocks. The final centroids are converted to the specification's `float` type,
and assignments are recomputed in Arrow batches against those exact centroids.
The assignment map records conversion, distance, and output worker time
alongside the end-to-end wall time for assignment, the required range shuffle,
partition-local sort, and Iceberg write. This makes the map cost visible without
pretending Spark can isolate the downstream shuffle and write behind one lazy
action.
Assigned postings are not persisted. Spark may therefore reevaluate assignment
while sampling range boundaries and writing the final data, trading additional
CPU work for bounded executor memory. Assignment counters report actual
evaluations and their ratio to source rows so this work and task retries remain
visible.

Postings are range partitioned and sorted by `(cid, key_1, key_2, ...)`.
Keeping `cid` first preserves cluster locality and enables file or row-group
pruning. Including every key column gives deterministic order and permits a
large cluster to split at a key boundary instead of forcing one `cid` into one
writer. The builder-neutral `WriteOptions` may request a positive output
partition count per build; when omitted, Relify uses the active Spark
context's default parallelism. Compression and target file size are published
as Iceberg write properties. Local row-count batching and row-group controls
belong to `Local`, not to the portable writer options.

PyIceberg creates the index tables with the canonical required-field schema in
the `relify` namespace, and Spark appends the distributed data through
DataFrameWriterV2. Relify publishes metadata only after both exact Iceberg
snapshots exist.

Spark queries are assembled with the public DataFrame API. Cluster routing is
a centroid Top-K followed by a postings left-semi join; large `nprobe` values
are not expanded into literal `IN` lists. Source resolution is omitted when
stored vectors and the requested projection permit an index-only query. The
terminal result is a native `pyspark.sql.DataFrame`.

The first implementation supports Spark Classic only. Spark Connect requires
a separate operation and cancellation lifecycle and is rejected explicitly.
SQLite is supported as a development, single-driver index catalog; a remote
catalog is required before production multi-driver use.

## Consequences

Local and Spark sessions share catalog and metadata semantics without sharing
execution code or runtime contexts. `VectorQuery` remains backend-independent,
while each session compiles it into its engine's native plan.

Spark and PyIceberg must be configured against the same Iceberg catalog under
the same logical name. A mismatched name, table UUID, or unavailable snapshot
fails before query execution.

Index construction is asynchronous and uses Spark job groups. The calling
Relify session owns status and waiting while `Spark(spark_session)` owns the
compute. Cross-driver build reservation, refresh, garbage collection of
Iceberg tables, Spark Connect, and remote index catalogs remain future work.

Spark's distributed sampling is Bernoulli rather than Faiss's in-process exact
sample without replacement, so its realized training-row count may differ
slightly from the 256-per-centroid target. It remains distributed and avoids a
global ordering or single-partition sample.

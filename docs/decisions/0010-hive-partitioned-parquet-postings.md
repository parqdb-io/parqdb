# Hive-Partitioned Parquet Postings

- Status: Accepted
- Date: 2026-08-03

## Context

An IVF query selects clusters before scanning Parquet postings. Keeping many
clusters in each Parquet file made this selection depend on row-group pruning,
which still caused near-full physical reads under a 2 GiB memory limit.

## Decision

The Parquet representation of `ivf_postings` uses Hive-style `cid` partitions
and stores each non-empty cluster in exactly one file. The physical Parquet file
omits `cid`; readers restore it as a required `INT32` partition column.
The exact-vector leaf uses Parquet `PLAIN` encoding with dictionary encoding
disabled. Scalar key columns retain the writer's default encoding choices.

The local builder hash-partitions rows by `cid` across a bounded number of
execution tasks and sorts each task by `cid`. A streaming writer keeps only the
current cluster file open for each task and closes it before advancing to the
next cluster. Hash collisions may share a build task but cannot share an output
file. The local reader lists an immutable postings relation once per session
and builds a `cid`-to-files manifest. Static `cid` predicates select files
directly from that manifest before DataFusion constructs the Parquet scan. The
scan still uses DataFusion's standard Parquet reader, projection, page cache,
and runtime-filter path; Relify does not cache decoded postings.

## Consequences

The number of postings files equals the number of non-empty clusters. A single
large cluster is not split according to the general target-file-size option.
Opening a relation performs one concurrent footer pass to populate DataFusion's
file-metadata cache; lazy provider creation charges this work to the first query.
Parquet readers must understand the defined Hive partition layout. This decision
does not change the Iceberg representation of IVF postings.

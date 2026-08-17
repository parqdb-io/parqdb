# Vendored arrow-rs Components

ParqDB vendors the `parquet` crate from arrow-rs 58.4.0 to maintain the narrow
Page-provider integration required by the Parquet Page cache. All other
arrow-rs crates continue to come from crates.io.

Changes from upstream must remain generic Parquet reader interfaces. ParqDB
cache policy, file identity, capacity management, and metrics belong in ParqDB
crates rather than this directory.

# Vendored arrow-rs Components

Relify vendors the `parquet` crate from arrow-rs 58.4.0 to maintain the narrow
Page-provider integration required by the Parquet Page cache. All other
arrow-rs crates continue to come from crates.io.

Changes from upstream must remain generic Parquet reader interfaces. Relify
cache policy, file identity, capacity management, and metrics belong in Relify
crates rather than this directory.

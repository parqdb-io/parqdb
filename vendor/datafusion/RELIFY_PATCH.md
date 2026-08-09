# Relify Patch

Upstream: Apache DataFusion 54.0.0, `datafusion-datasource-parquet` crate.

Relify adds a generic `ParquetPageCacheFactory` session extension. The Parquet
source creates one file-scoped cache handle for each `PartitionedFile` and
passes it to arrow-rs `ParquetPushDecoder`. The extension contains no Relify
index or query semantics.

When upgrading:

1. replace this directory with the published upstream crate;
2. reapply the factory extension and its propagation to the push decoder;
3. run DataFusion Parquet tests and Relify's normal `register_parquet` scan
   tests; and
4. verify object-store URL and file metadata reach the factory unchanged.

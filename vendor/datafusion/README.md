# Vendored DataFusion Components

Relify vendors `datafusion-datasource-parquet` 54.0.0 to expose a generic
file-scoped Parquet Page-cache factory at the push-decoder boundary. The
extension contains no Relify index or query semantics.

The rest of DataFusion remains supplied by crates.io.

# Third-Party Notices

Relify's original source code is licensed under the MIT License. The Python
distribution also contains modified source and compiled code from
[Apache DataFusion Python](https://github.com/apache/datafusion-python), version
54.0.0; the Apache Arrow Rust `parquet` crate, version 58.4.0; and the Apache
DataFusion `datafusion-datasource-parquet` crate, version 54.0.0. These
components are licensed under the Apache License 2.0.
The DataFusion backend also compiles exact-snapshot table support from
[Apache Iceberg Rust](https://github.com/apache/iceberg-rust), pinned in
`Cargo.lock` and licensed under the Apache License 2.0.

The complete Apache License 2.0 text is distributed in
`vendor/datafusion-python/LICENSE.txt`,
`vendor/arrow-rs/parquet/LICENSE.txt`, and
`vendor/datafusion/datasource-parquet/LICENSE.txt`, and in every wheel's
standard `.dist-info/licenses/` directory. Relify's maintained patches are
documented under the corresponding vendor directories.

Every wheel contains a CycloneDX SBOM under `.dist-info/sboms/` with the locked
Rust dependency graph, component versions, declared licenses, package URLs, and
available registry checksums.

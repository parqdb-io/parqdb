# Third-Party Notices

Relify's original source code is licensed under the MIT License. The Python
distribution also contains modified source and compiled code from
[Apache DataFusion Python](https://github.com/apache/datafusion-python), version
54.0.0, licensed under the Apache License 2.0.
The DataFusion backend also compiles exact-snapshot table support from
[Apache Iceberg Rust](https://github.com/apache/iceberg-rust), pinned in
`Cargo.lock` and licensed under the Apache License 2.0.

The complete Apache License 2.0 text is distributed in
`vendor/datafusion-python/LICENSE.txt` and in every wheel's standard
`.dist-info/licenses/` directory. Relify's maintained patch is documented in
`vendor/datafusion-python/RELIFY_PATCH.md`.

Every wheel contains a CycloneDX SBOM under `.dist-info/sboms/` with the locked
Rust dependency graph, component versions, declared licenses, package URLs, and
available registry checksums.

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

The in-memory HNSW centroid navigator adapts graph construction and search
techniques from [Faiss](https://github.com/facebookresearch/faiss), reference
commit `1f93154314afbef210f0ebebeab840da22f9ec7d`, licensed under the MIT License.

## Faiss MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The complete Apache License 2.0 text is distributed in
`vendor/datafusion-python/LICENSE.txt`,
`vendor/arrow-rs/parquet/LICENSE.txt`, and
`vendor/datafusion/datasource-parquet/LICENSE.txt`, and in every wheel's
standard `.dist-info/licenses/` directory. Relify's maintained patches are
documented under the corresponding vendor directories.

Every wheel contains a CycloneDX SBOM under `.dist-info/sboms/` with the locked
Rust dependency graph, component versions, declared licenses, package URLs, and
available registry checksums.

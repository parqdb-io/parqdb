# ParqDB Patch

Upstream: Apache arrow-rs 58.4.0, `parquet` crate.

ParqDB adds a generic, file-scoped `PageCache` hook to the standard Parquet
Page readers and propagates it through the synchronous, asynchronous, and push
decoder builders. A completely cached requested range can bypass storage I/O;
all values still pass through the upstream Arrow decoders.

The vendored crate must not contain ParqDB index, query, capacity, eviction, or
file-identity policy. Those concerns remain in `parqdb-local`.

When upgrading:

1. replace this directory with the published upstream `parquet` crate;
2. reapply only the generic Page-cache interface and propagation;
3. run the vendored crate tests and ParqDB's warm-read tests; and
4. verify that a warm push-decoder scan requests no column-chunk ranges.

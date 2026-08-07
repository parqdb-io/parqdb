# Full-Text Search Prototypes

These programs test the unresolved storage and execution choices described in
[`docs/design/full-text-search.md`](../../../docs/design/full-text-search.md).
They are isolated experiments, not Relify APIs or published index formats.

## Posting Prototype

`prototype.py` builds a synthetic text source, row-per-posting and nested-list
posting-block Parquet indexes, and a Tantivy baseline. It runs rare and common
term and phrase queries, checks both Parquet layouts against an exact source
scan, and records logical I/O.

```bash
uv run benchmarks/tools/full_text_search/prototype.py \
  --root /tmp/relify-full-text \
  --rebuild \
  --output /tmp/relify-full-text/result.json
```

The comparison is a feasibility test. Parquet query execution is Python while
Tantivy executes native Rust, so the latency numbers do not isolate the storage
formats. The nested-list layout is a deliberately simple baseline; it is not a
proposed Relify index format.

## Payload Storage

The standalone Rust crate measures three related questions:

- `page-size`: random Top-K reads with 16, 64, 256, and 1,024 KiB Parquet data
  page targets;
- `large-text`: Parquet page compression versus independent Zstandard frames
  for 1 MiB or larger text values; and
- `datafusion-payload`: decoded strings through a DataFusion Top-K plan versus
  binary Zstandard frames decoded by a UDF after Top-K.

Generate the 1 MiB payload files:

```bash
cargo run --release \
  --manifest-path benchmarks/tools/full_text_search/storage/Cargo.toml \
  --bin large-text -- \
  --text-mib 1 --rows 128 --queries 30 --warmups 5 --top-k 10 \
  --output-dir /tmp/relify-large-text-1m
```

Run each DataFusion mode in a separate process so Linux peak RSS is independent:

```bash
cargo run --release \
  --manifest-path benchmarks/tools/full_text_search/storage/Cargo.toml \
  --bin datafusion-payload -- \
  --mode page --path /tmp/relify-large-text-1m/page-zstd-batch-1.parquet \
  --queries 10 --warmups 2 --batch-size 8 \
  --output /tmp/relify-datafusion-1m-page.json

cargo run --release \
  --manifest-path benchmarks/tools/full_text_search/storage/Cargo.toml \
  --bin datafusion-payload -- \
  --mode row --path /tmp/relify-large-text-1m/row-zstd.parquet \
  --queries 10 --warmups 2 --batch-size 8 \
  --output /tmp/relify-datafusion-1m-row.json
```

The RSS measurement reads Linux `/proc/self/status`; the DataFusion payload
binary therefore requires Linux. All measurements use synthetic data and a
warm local page cache unless the caller controls the cache externally.

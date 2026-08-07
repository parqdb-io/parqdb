# Quantized Code Layout Benchmark

This standalone benchmark compares three candidate Parquet layouts for scalar-
quantized IVF postings:

- `columns`: SQ8 and LVQ8 use one required `UINT8` column per dimension. SQ4
  and LVQ4 use one required `UINT8` column per pair of dimensions, with two
  codes packed into each byte. Distance evaluation scans each column
  contiguously and accumulates into one `FLOAT32` distance buffer.
- `list`: one required Parquet `LIST<UINT8>` column, matching the nested layout
  used by the current IVF-Flat vector column. SQ4 and LVQ4 store two codes per
  byte; SQ8 and LVQ8 store one code per byte. Distance evaluation processes one
  vector at a time.
- `fixed_binary`: one required Parquet `FIXED_LEN_BYTE_ARRAY` column whose
  length is fixed by the quantizer and vector dimension. It uses the same
  row-wise distance kernel as `list` without nested LIST decoding.

SQ4 and SQ8 use global per-dimension offset and scale parameters. LVQ4 and LVQ8
quantize each original vector with a per-row offset and scale. All layouts use
the same quantized codes, so layout does not affect search quality.

Both distance paths use the same runtime-selected SIMD backend. The benchmark
reports:

- in-memory distance throughput at several candidate-set sizes, processed as
  8,192-row DataFusion-sized batches;
- warm-cache Parquet scan and decode time;
- warm-cache Parquet scan, decode, and distance time;
- persisted file size, resident code bytes, distance-buffer bytes, and maximum
  decoded Arrow batch bytes.

Input loading, quantizer training, encoding, Parquet writing, and correctness
validation are outside measured sections. Parquet files are uncompressed, use
PLAIN data encoding with dictionary encoding disabled, contain 8,192-row row
groups by default, and are read in 8,192-row batches.

Parquet has no physical 8-bit integer type. The `UINT8` columns and LIST
elements use physical `INT32` with an unsigned 8-bit logical annotation, while
`fixed_binary` uses physical `FIXED_LEN_BYTE_ARRAY`. The result file records
this distinction explicitly.

Run against any `fvecs` file whose source dimension is at least 384:

```bash
cargo run --release -- \
  --input /path/to/base.fvecs \
  --rows 262144 \
  --dimension 384 \
  --candidate-rows 8192,65536,262144
```

Candidate-set size changes total work but not the execution batch or distance-
buffer size. The generated Parquet files and `result.json` are written under
`target/quantized-layout` unless `--output-dir` or `--output` is specified.

## Decision

Use one required `FIXED_LEN_BYTE_ARRAY(code_size)` column for quantized codes.
The scalar-column and LIST layouts remain benchmark controls and are not index-
schema candidates.

The deciding run used 262,144 vectors, 384 dimensions, 8,192-row batches,
AVX-512 distance kernels, uncompressed PLAIN encoding, and ten measured runs
after three warmups. Input preparation and Parquet writing were not timed.

| Quantizer | Layout | File MiB | Scan + distance p50 ms |
| --- | --- | ---: | ---: |
| SQ4 | columns | 192.9 | 79.0 |
| SQ4 | list | 193.3 | 237.4 |
| SQ4 | fixed binary | 48.0 | 20.9 |
| SQ8 | columns | 385.8 | 154.4 |
| SQ8 | list | 385.3 | 529.4 |
| SQ8 | fixed binary | 96.0 | 40.5 |
| LVQ4 | columns | 194.9 | 96.0 |
| LVQ4 | list | 195.3 | 249.1 |
| LVQ4 | fixed binary | 50.0 | 21.7 |
| LVQ8 | columns | 387.8 | 167.7 |
| LVQ8 | list | 387.3 | 482.0 |
| LVQ8 | fixed binary | 98.0 | 34.5 |

The `UINT8` columns and LIST elements use Parquet physical `INT32`, so PLAIN
encoding stores four bytes per code byte. Fixed binary stores the packed bytes
directly and avoids both nested LIST decoding and a 192- or 384-column schema.

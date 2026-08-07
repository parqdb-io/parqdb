# Full-Text Search

This document records the current design direction for full-text search in
Relify. It is an exploration, not a description of a released API or index
format.

## Goal

Relify should support full-text retrieval over append-oriented lakehouse data,
with observability traces, RAG documents, and model inputs and outputs as the
initial workloads. A query must be able to combine:

- structured predicates such as tenant, time range, status, and model;
- BM25 ranking;
- exact phrase filtering; and
- Top-K retrieval of selected source fields.

The first implementation targets the embedded DataFusion backend. Extending the
format or execution path to other engines is not an initial requirement.

Real-time row updates, fuzzy or prefix search, highlighting, advanced dynamic
pruning such as WAND, and Lucene-style segment merging are outside the first
implementation.

## Query Path

The intended execution path is:

1. Analyze the query with the analyzer recorded by the index.
2. Locate only the posting partitions and row groups for the query terms.
3. Evaluate term or phrase matches and compute BM25 scores.
4. Retain the Top-K `(doc_id, score)` rows.
5. Fetch requested source columns for those document IDs.
6. Decompress large text payloads only if the final projection needs them.

The public API is not fixed. The SQL shape should nevertheless remain
recognizable as a ranked relation that can be filtered, joined, and aggregated:

```sql
SELECT
    trace_id,
    timestamp,
    level,
    text,
    bm25_score(text, 'request failed') AS _score
FROM traces
WHERE tenant_id = 42
  AND timestamp >= TIMESTAMP '2026-08-01 00:00:00'
  AND phrase_match(text, 'request failed')
ORDER BY _score DESC
LIMIT 10;
```

`bm25_score` and `phrase_match` are illustrative names, not committed SQL
extensions.

## Index Model

The prototype separates four logical relations:

| Relation | Purpose |
| --- | --- |
| Index metadata | Analyzer configuration, indexed source field, BM25 parameters, document count, and total document length |
| Terms | One row per term with its document frequency and posting location |
| Documents | One row per document with its length and source locator |
| Postings | Term, document ID, term frequency, and optional positions |

The relations use Parquet-compatible scalar, list, and binary values. Posting
files are partitioned by a stable term hash and ordered by `(term, doc_id)` so a
query can select files from the term hash and row groups from term statistics.
Positions are read only for phrase queries.

The current row-per-posting representation is a correctness baseline, not a
format decision. A second prototype grouped postings into Parquet `LIST`
columns. It reduced the number of encoded rows but increased bytes read and did
not improve phrase queries. Generic nested lists are therefore not the planned
block representation.

The next storage experiment may encode document deltas, term frequencies, and
positions into compact binary posting blocks. That representation is
acceptable only if its encoding is documented, versioned, independently
decodable, and still stored in ordinary Parquet tables. It must demonstrate a
material advantage over row-per-posting Parquet before becoming a format
proposal.

## Large Text Payloads

Large model inputs and outputs must not travel through the scoring and Top-K
pipeline as decoded Arrow strings. For Relify-managed data, the current
direction is:

- compress each large UTF-8 payload independently as a standard Zstandard
  frame;
- store the frame in a Parquet `BYTE_ARRAY` column with Parquet compression
  disabled for that column;
- record the payload codec and logical content type in table metadata;
- keep structured fields as native Parquet columns; and
- resolve and decompress payloads after Top-K unless an earlier text operation
  explicitly needs their contents.

Independent frames bound decompression to the selected rows. The physical
writer should submit at most one large payload per Parquet write batch so one
oversized value does not cause unrelated rows to share a very large data page.
This is a writer requirement, not part of the logical full-text index schema.

The optimized DataFusion path will require a late source lookup. A logical
`doc_id` remains the query identity; an index snapshot may additionally record
physical `(file, row_group, row_offset)` locators for a `PayloadFetchExec` to
issue precise reads. Generic engines can still join by `doc_id`, but will not
receive the same random-read optimization automatically.

## Current Evidence

The feasibility prototype generated 20,000 documents with 160 tokens each and
compared row-per-posting Parquet, nested-list posting blocks, and Tantivy. Both
Parquet layouts produced the exact same ordered BM25 and phrase results as a
source scan. The following warm-cache timings cover only index lookup, scoring,
and Top-K:

| Query | Parquet rows | Parquet nested blocks | Tantivy |
| --- | ---: | ---: | ---: |
| Rare term | 4.43 ms | 8.27 ms | 0.05 ms |
| Common term | 36.09 ms | 32.15 ms | 0.24 ms |
| Rare phrase | 94.90 ms | 126.66 ms | 0.12 ms |
| Common phrase | 309.31 ms | 326.60 ms | 4.45 ms |

The row layout stored 1,919,895 posting rows in 7.81 MB. Grouping 256 documents
per nested block reduced that to 13,294 Parquet rows, but increased the index to
8.08 MB. Tantivy used 5.91 MB. The nested layout selected fewer row groups but
read more bytes for every query in this workload.

The Parquet prototype is Python and Tantivy executes native Rust, so these are
not implementation-level comparative benchmarks. They establish that term
partitioning and row-group pruning work, while the row-per-posting execution
path is not yet competitive for common terms or positions. It also shows that
Parquet row count is not the controlling variable: a useful posting block needs
specialized compression and native execution, not merely nested arrays.

A separate DataFusion experiment compared ordinary Parquet-compressed strings
with independently compressed payloads decoded after Top-K:

| Payload per row | Parquet string p50 | Late decode p50 | Speedup | Peak RSS reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 MiB | 158.71 ms | 20.47 ms | 7.75x | 57% |
| 10 MiB | 807.07 ms | 246.00 ms | 3.28x | 31% |

Both representations had effectively identical file sizes on the synthetic
corpus. The experiment used a warm local page cache and still scanned the
compressed payload column before Top-K. A physical late-fetch operator should
remove that remaining scan and must be measured separately.

## Implementation Stages

1. **Storage baseline.** Keep the current correctness prototype and payload
   experiments reproducible. Treat nested-list blocks as a rejected baseline.
   Add scale, selectivity, index-size, and remote range-read measurements for a
   compact binary posting codec before selecting a layout.
2. **Native index execution.** Implement term lookup, posting decode, phrase
   evaluation, BM25, and Top-K in Rust for the embedded backend. Compare
   row-per-posting and block layouts without Python object conversion.
3. **Late source fetch.** Add a provider or physical operator that accepts
   selected document locators and reads only the required Parquet pages. Decode
   independently compressed payloads after this lookup.
4. **Integrated write and query path.** Build the full-text index and payload
   layout in the same publication transaction, expose one relational search
   API, and validate snapshot consistency and refresh behavior.

The first implementation is complete only when it returns the same ordered
results as the reference path, demonstrates bounded memory for large payloads,
and reports physical bytes and range requests for storage-backed queries.

# Format Version 1 Fixtures

These fixtures are shared test vectors for Relify metadata and IVF query
semantics. They are non-normative; the specification remains authoritative.

`valid/` contains two complete logical Parquet indexes. The root fixture uses
one string source key and stores exact vectors in postings:

- `metadata.json`: portable Relify metadata;
- `metadata-v2.json`: a legal immutable update from `metadata.json`;
- `source.parquet`: the indexed source table;
- `ivf_centroids.parquet`: the centroid relation;
- `ivf_postings/cid=<value>/part-0.parquet`: the Hive-partitioned posting
  relation; and
- `queries.json`: inputs and ordered expected results.

`valid/composite_no_vectors/` uses the same file layout with two ordered source
key fields (`int`, then `string`) and `store_vectors = false`. Its postings omit
the `vector` field, so candidate distance evaluation must resolve vectors from
the source table. Its query cases cover composite-key ordering.

The URIs in `metadata.json` are stable logical fixture URIs. A test harness maps
them to the files in `valid/`; it must not rewrite the metadata document.

`catalog.json` is an ordered catalog operation trace. It covers absent loads,
registration, duplicate registration, compare-and-swap publication, a stale
commit conflict, and drop. Metadata and location names in the trace are keys
into its `metadata` and `locations` maps; the fixture does not prescribe a
catalog API or protocol representation.

`invalid/` contains metadata documents that readers must reject.
`invalid/manifest.json` enumerates every document and the violated invariant.
Harnesses should consume the manifest rather than maintaining a private file
list.

The valid query cases cover cluster-ID tie-breaking, equal-distance results,
source prefilters, full probes, `k` larger than the selected candidate set, and
empty results. Result order within an equal-distance group is not significant.
Query `filter` objects are fixture-harness inputs, not a portable filter syntax.

Relify's DuckDB interoperability test reads these Parquet files and reproduces
all expected results without importing the Relify package:

```bash
make test-interop
```

Run the generator after intentionally changing the fixture:

```bash
uv run python spec/fixtures/v1/generate.py
```

The generator is deterministic at the logical Arrow and JSON level. Parquet
bytes may differ across PyArrow releases, so fixture review should compare
schemas and values rather than file hashes.

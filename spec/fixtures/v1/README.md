# Format Version 1 Fixtures

These non-normative fixtures exercise Relify metadata and IVF schema version
1. The specification remains authoritative.

`valid/` contains a source-encoded IVF fixture. It includes logical index
metadata, IVF-centroids metadata, source and centroid Parquet files,
Hive-partitioned postings, and ordered query results.

Additional directories cover:

- `valid/composite/`: a composite source key;
- `valid/lvq4/`: LVQ4 codes and expected approximate distances; and
- `valid/lvq8/`: LVQ8 codes and expected approximate distances.

All logical indexes use the same public IVF schema version. Encoding changes
the postings fields, not the schema-version history. Full source vectors are
absent from every postings fixture.

The metadata URIs are stable logical fixture URIs. Test harnesses may map them
to local files but must preserve the referenced relation state and IVF-centroids
descriptor.

`catalog.json` is an ordered catalog operation trace covering registration,
duplicate registration, compare-and-swap publication, stale commits, and drop.

`invalid/manifest.json` enumerates every invalid metadata document and the
invariant it violates. Harnesses should consume the manifest instead of
maintaining a separate list.

The DuckDB interoperability test reproduces source-encoded query results
without importing Relify:

```bash
make test-interop
```

Regenerate fixtures after an intentional schema change:

```bash
.venv/bin/python spec/fixtures/v1/generate.py
```

Generation is deterministic at the JSON and Arrow value level. Parquet bytes
may vary across PyArrow releases.

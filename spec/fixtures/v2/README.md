# IVF Schema v2 Fixtures

These fixtures define portable LVQ4 and LVQ8 examples for IVF schema version
2. Each directory contains metadata, source and centroid tables, partitioned
postings, and expected query results.

The source vectors have dimension 3 so the LVQ4 fixture also exercises nibble
ordering and the required zero high nibble in the final byte. For source vector
`[0.0, 0.5, 1.0]`, the encoded bytes are:

| Encoding | Codes | Stored bytes |
|---|---|---|
| `lvq4` | `[0, 8, 15]` | `80 0f` |
| `lvq8` | `[0, 128, 255]` | `00 80 ff` |

Regenerate the fixtures from the repository root:

```bash
uv run python spec/fixtures/v2/generate.py
```

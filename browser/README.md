# @parqdb/browser

Browser client for immutable ParqDB IVF-LVQ4/LVQ8 packages hosted on public
HTTPS object storage.

```ts
import { ParqDB } from '@parqdb/browser'

const index = await ParqDB.open('https://example.com/index/manifest.json')
const hits = await index.search(query, { nprobe: 64, k: 10 })
```

The client fetches the exact package manifest, ranks all leaf centroids, reads
only postings files and row groups intersecting the selected CIDs, and returns
the source-key fields plus `_distance`. It never lists the object prefix and
does not fetch or join the source table. Selected row-group reads run with a
bounded concurrency of eight by default; set `maxConcurrentReads` in the search
options to a value from 1 through 64 to tune the transport. Concurrent slices
of the same object are coalesced into fewer HTTP requests when their total gaps
fit within 64 KiB. Set `maxRangeGapBytes` when opening the index to tune that
bounded over-read, or set it to zero to merge only overlapping and adjacent
ranges.

Partial object reads also use a 32 MiB in-memory cache shared by all postings
files opened by one `ParqDB` instance. The cache stores immutable, 256 KiB
aligned chunks, deduplicates in-flight loads, and evicts least-recently-used
chunks at the byte budget. A cache hit does not call `fetch`. Set
`rangeCacheBytes` when opening the index to change the budget, or set it to zero
to disable this cache. The underlying partial requests continue to use
`cache: 'no-store'` so correctness does not depend on browser-specific caching
of `206 Partial Content` responses.

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
options to a value from 1 through 64 to tune the transport.

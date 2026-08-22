import type { AsyncBuffer } from 'hyparquet'

export interface HttpOptions {
  signal?: AbortSignal
  allowHttp?: boolean
  fetch?: typeof globalThis.fetch
  maxManifestBytes?: number
  maxRangeBytes?: number
  maxRangeGapBytes?: number
  /** Byte budget for the shared in-memory HTTP Range cache. Set to zero to disable it. */
  rangeCacheBytes?: number
  /** Share one cache across multiple immutable objects or ParqDB clients. */
  rangeCache?: HttpRangeCache
}

export interface HttpRangeCacheStats {
  capacityBytes: number
  sizeBytes: number
  entries: number
  hits: number
  misses: number
  evictions: number
}

interface CachedRange {
  key: string
  objectKey: string
  start: number
  end: number
  bytes: Promise<ArrayBuffer>
  settled: boolean
  lastUsed: number
}

const DEFAULT_RANGE_CACHE_BYTES = 32 * 1024 * 1024

/** A byte-bounded LRU of exact ranges from immutable HTTP objects. */
export class HttpRangeCache {
  private readonly ranges = new Map<string, CachedRange>()
  private sizeBytes = 0
  private clock = 0
  private hitCount = 0
  private missCount = 0
  private evictionCount = 0

  constructor(readonly capacityBytes = DEFAULT_RANGE_CACHE_BYTES) {
    if (!Number.isSafeInteger(capacityBytes) || capacityBytes < 0) {
      throw new RangeError('rangeCacheBytes must be a non-negative portable integer')
    }
  }

  stats(): HttpRangeCacheStats {
    return {
      capacityBytes: this.capacityBytes,
      sizeBytes: this.sizeBytes,
      entries: this.ranges.size,
      hits: this.hitCount,
      misses: this.missCount,
      evictions: this.evictionCount,
    }
  }

  async read(
    url: URL,
    objectSize: number,
    start: number,
    end: number,
    load: (start: number, end: number) => Promise<ArrayBuffer>,
  ): Promise<ArrayBuffer> {
    if (this.capacityBytes === 0) return load(start, end)
    const objectKey = `${url.href}\u0000${objectSize}`
    let covering: CachedRange | undefined
    for (const entry of this.ranges.values()) {
      if (
        entry.objectKey === objectKey &&
        entry.start <= start &&
        entry.end >= end &&
        (covering === undefined || entry.end - entry.start < covering.end - covering.start)
      ) covering = entry
    }
    if (covering !== undefined) {
      this.hitCount += 1
      covering.lastUsed = ++this.clock
      const bytes = await covering.bytes
      return bytes.slice(start - covering.start, end - covering.start)
    }
    this.missCount += 1
    const length = end - start
    if (length > this.capacityBytes) return load(start, end)

    const key = `${objectKey}\u0000${start}\u0000${end}`
    const entry: CachedRange = {
      key,
      objectKey,
      start,
      end,
      bytes: Promise.resolve().then(() => load(start, end)),
      settled: false,
      lastUsed: ++this.clock,
    }
    this.ranges.set(key, entry)
    this.sizeBytes += length
    void entry.bytes.then(
      () => {
        if (this.ranges.get(key) !== entry) return
        entry.settled = true
        for (const candidate of this.ranges.values()) {
          if (
            candidate !== entry && candidate.settled &&
            candidate.objectKey === objectKey &&
            candidate.start >= start && candidate.end <= end
          ) this.remove(candidate)
        }
        this.evict()
      },
      () => this.remove(entry),
    )
    return entry.bytes
  }

  private evict(): void {
    while (this.sizeBytes > this.capacityBytes) {
      let oldest: CachedRange | undefined
      for (const entry of this.ranges.values()) {
        if (entry.settled && (oldest === undefined || entry.lastUsed < oldest.lastUsed)) oldest = entry
      }
      if (oldest === undefined) return
      this.remove(oldest)
      this.evictionCount += 1
    }
  }

  private remove(entry: CachedRange): void {
    if (this.ranges.get(entry.key) !== entry) return
    this.ranges.delete(entry.key)
    this.sizeBytes -= entry.end - entry.start
  }

}

interface PendingRange {
  start: number
  end: number
  resolve: (bytes: ArrayBuffer) => void
  reject: (error: unknown) => void
}

interface CoalescedRange {
  start: number
  end: number
  gapBytes: number
  reads: PendingRange[]
}

export async function fetchManifest(url: URL, options: HttpOptions): Promise<ArrayBuffer> {
  validateUrl(url, options.allowHttp ?? false)
  const fetcher = options.fetch ?? globalThis.fetch
  if (fetcher === undefined) throw new Error('ParqDB browser client requires fetch')
  const request: RequestInit = {
    cache: 'force-cache',
    credentials: 'omit',
    redirect: 'follow',
  }
  if (options.signal !== undefined) request.signal = options.signal
  const response = await fetcher(url, request)
  validateResponseUrl(response, options.allowHttp ?? false)
  if (!response.ok) throw new Error(`manifest request failed with HTTP ${response.status}`)
  const bytes = await response.arrayBuffer()
  const limit = options.maxManifestBytes ?? 8 * 1024 * 1024
  if (bytes.byteLength === 0 || bytes.byteLength > limit) {
    throw new Error(`manifest size must be in [1, ${limit}] bytes`)
  }
  return bytes
}

export class HttpRangeBuffer implements AsyncBuffer {
  readonly byteLength: number
  private pending: PendingRange[] = []
  private flushScheduled = false
  private readonly rangeCache: HttpRangeCache

  constructor(
    private readonly url: URL,
    byteLength: number,
    private readonly options: HttpOptions,
  ) {
    validateUrl(url, options.allowHttp ?? false)
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
      throw new Error('HTTP object size must be a positive portable integer')
    }
    this.byteLength = byteLength
    this.rangeCache = options.rangeCache ?? new HttpRangeCache(options.rangeCacheBytes)
  }

  async slice(start: number, end = this.byteLength): Promise<ArrayBuffer> {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end <= start || end > this.byteLength) {
      throw new RangeError(`invalid HTTP byte range [${start}, ${end}) for ${this.byteLength}-byte object`)
    }
    const maxRangeBytes = this.options.maxRangeBytes ?? 64 * 1024 * 1024
    if (end - start > maxRangeBytes && !(start === 0 && end === this.byteLength)) {
      throw new RangeError(`HTTP byte range exceeds ${maxRangeBytes}-byte client limit`)
    }
    if (start === 0 && end === this.byteLength) return this.fetchSlice(start, end)
    return new Promise<ArrayBuffer>((resolve, reject) => {
      this.pending.push({ start, end, resolve, reject })
      if (this.flushScheduled) return
      this.flushScheduled = true
      setTimeout(() => {
        this.flushScheduled = false
        const pending = this.pending
        this.pending = []
        void this.flush(pending)
      }, 0)
    })
  }

  private async flush(pending: PendingRange[]): Promise<void> {
    const maxRangeBytes = this.options.maxRangeBytes ?? 64 * 1024 * 1024
    const maxRangeGapBytes = this.options.maxRangeGapBytes ?? 64 * 1024
    if (!Number.isSafeInteger(maxRangeGapBytes) || maxRangeGapBytes < 0 || maxRangeGapBytes > maxRangeBytes) {
      const error = new RangeError(`maxRangeGapBytes must be a portable integer in [0, ${maxRangeBytes}]`)
      pending.forEach(read => read.reject(error))
      return
    }
    const coalesced: CoalescedRange[] = []
    for (const read of pending.sort((left, right) => left.start - right.start || left.end - right.end)) {
      const previous = coalesced.at(-1)
      const gap = previous === undefined ? 0 : Math.max(0, read.start - previous.end)
      const mergedEnd = previous === undefined ? read.end : Math.max(previous.end, read.end)
      if (
        previous !== undefined &&
        previous.gapBytes + gap <= maxRangeGapBytes &&
        mergedEnd - previous.start <= maxRangeBytes
      ) {
        previous.end = mergedEnd
        previous.gapBytes += gap
        previous.reads.push(read)
      } else {
        coalesced.push({ start: read.start, end: read.end, gapBytes: 0, reads: [read] })
      }
    }
    await Promise.all(coalesced.map(async range => {
      try {
        const bytes = await this.rangeCache.read(
          this.url,
          this.byteLength,
          range.start,
          range.end,
          (start, end) => this.fetchSlice(start, end, true),
        )
        for (const read of range.reads) {
          read.resolve(bytes.slice(read.start - range.start, read.end - range.start))
        }
      } catch (error) {
        range.reads.forEach(read => read.reject(error))
      }
    }))
  }

  private async fetchSlice(start: number, end: number, forceRange = false): Promise<ArrayBuffer> {
    const fetcher = this.options.fetch ?? globalThis.fetch
    if (fetcher === undefined) throw new Error('ParqDB browser client requires fetch')
    const complete = !forceRange && start === 0 && end === this.byteLength
    const request: RequestInit = {
      // Browsers are allowed to cache partial responses, but some intermediary and
      // browser-cache combinations have incorrectly reused a different cached 206.
      // The Parquet reader already caches metadata and centroids above this layer.
      cache: complete ? 'force-cache' : 'no-store',
      credentials: 'omit',
      redirect: 'follow',
    }
    if (!complete) request.headers = { Range: `bytes=${start}-${end - 1}` }
    if (this.options.signal !== undefined) request.signal = this.options.signal
    const response = await fetcher(this.url, request)
    validateResponseUrl(response, this.options.allowHttp ?? false)
    if (complete && response.status === 200) {
      const bytes = await response.arrayBuffer()
      if (bytes.byteLength !== this.byteLength) throw new Error('complete HTTP object length does not match manifest')
      return bytes
    }
    if (response.status !== 206) {
      throw new Error(`range request must return 206, received ${response.status}`)
    }
    validateContentRange(response.headers.get('content-range'), start, end - 1, this.byteLength)
    const bytes = await response.arrayBuffer()
    if (bytes.byteLength !== end - start) throw new Error('range response length does not match Content-Range')
    return bytes
  }
}

function validateContentRange(header: string | null, start: number, end: number, size: number): void {
  const expected = `bytes ${start}-${end}/${size}`
  if (header === null) {
    throw new Error(
      `missing Content-Range; expected ${expected}. For cross-origin indexes, expose it with Access-Control-Expose-Headers`,
    )
  }
  const match = /^bytes\s+([0-9]+)-([0-9]+)\/([0-9]+)$/i.exec(header.trim())
  if (
    match === null ||
    Number(match[1]) !== start ||
    Number(match[2]) !== end ||
    Number(match[3]) !== size
  ) {
    throw new Error(`invalid Content-Range ${JSON.stringify(header)}; expected ${expected}`)
  }
}

export function objectUrl(manifestUrl: URL, path: string): URL {
  const base = new URL('.', manifestUrl)
  const resolved = new URL(path, base)
  if (!resolved.href.startsWith(base.href)) throw new Error('artifact object URL escapes publication root')
  return resolved
}

function validateUrl(url: URL, allowHttp: boolean): void {
  if (url.username !== '' || url.password !== '' || url.search !== '' || url.hash !== '') {
    throw new Error('artifact URLs cannot contain credentials, query, or fragment')
  }
  if (url.protocol === 'https:') return
  if (allowHttp && url.protocol === 'http:') return
  throw new Error('artifact URLs must use HTTPS')
}

function validateResponseUrl(response: Response, allowHttp: boolean): void {
  if (response.url === '') return
  const url = new URL(response.url)
  if (url.protocol === 'https:' || (allowHttp && url.protocol === 'http:')) return
  throw new Error('final artifact response URL must use HTTPS')
}

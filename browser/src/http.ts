import type { AsyncBuffer } from 'hyparquet'

export interface HttpOptions {
  signal?: AbortSignal
  allowHttp?: boolean
  fetch?: typeof globalThis.fetch
  maxManifestBytes?: number
  maxRangeBytes?: number
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
  }

  async slice(start: number, end = this.byteLength): Promise<ArrayBuffer> {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end <= start || end > this.byteLength) {
      throw new RangeError(`invalid HTTP byte range [${start}, ${end}) for ${this.byteLength}-byte object`)
    }
    const maxRangeBytes = this.options.maxRangeBytes ?? 64 * 1024 * 1024
    if (end - start > maxRangeBytes && !(start === 0 && end === this.byteLength)) {
      throw new RangeError(`HTTP byte range exceeds ${maxRangeBytes}-byte client limit`)
    }
    const fetcher = this.options.fetch ?? globalThis.fetch
    if (fetcher === undefined) throw new Error('ParqDB browser client requires fetch')
    const complete = start === 0 && end === this.byteLength
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
  if (!resolved.href.startsWith(base.href)) throw new Error('package object URL escapes package root')
  return resolved
}

function validateUrl(url: URL, allowHttp: boolean): void {
  if (url.username !== '' || url.password !== '' || url.search !== '' || url.hash !== '') {
    throw new Error('package URLs cannot contain credentials, query, or fragment')
  }
  if (url.protocol === 'https:') return
  if (allowHttp && url.protocol === 'http:') return
  throw new Error('package URLs must use HTTPS')
}

function validateResponseUrl(response: Response, allowHttp: boolean): void {
  if (response.url === '') return
  const url = new URL(response.url)
  if (url.protocol === 'https:' || (allowHttp && url.protocol === 'http:')) return
  throw new Error('final package response URL must use HTTPS')
}

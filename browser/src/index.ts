import { parquetMetadataAsync, parquetReadObjects } from 'hyparquet'
import type { FileMetaData, RowGroup } from 'hyparquet'
import { compressors } from 'hyparquet-compressors'

import { HttpRangeBuffer, fetchManifest, objectUrl } from './http.js'
import type { HttpOptions } from './http.js'
import { WasmKernel } from './kernel.js'
import type { WasmSource } from './kernel.js'
import { parseManifest } from './manifest.js'
import type { PackageManifest, PostingsFile, SourceKeyField } from './manifest.js'

export type SourceKeyValue = boolean | number | bigint | string | Date | Uint8Array
export type SearchHit = Record<string, SourceKeyValue | number> & { _distance: number }

export interface OpenOptions extends HttpOptions {
  wasm?: WasmSource
  maxObjects?: number
  maxDimension?: number
  maxNlist?: number
}

export interface SearchOptions {
  nprobe: number
  k: number
  signal?: AbortSignal
  maxCandidateRows?: number
  trace?: (event: QueryTraceEvent) => void
}

export interface QueryTraceEvent {
  phase: 'routing' | 'planning' | 'scoring'
  status: 'start' | 'complete'
  selectedCids?: number
  files?: number
  candidateRows?: number
}

interface Candidate {
  row: Record<string, unknown>
  distance: number
  order: number
}

interface RowSpan {
  start: number
  end: number
}

export class ParqDB {
  readonly manifest: PackageManifest
  private centroidData: Promise<{
    codes: Uint8Array
    offsets: Float32Array
    scales: Float32Array
    cids: Int32Array
  }> | undefined
  private readonly postingsMetadata = new Map<string, Promise<FileMetaData>>()

  private constructor(
    manifest: PackageManifest,
    private readonly manifestUrl: URL,
    private readonly kernel: WasmKernel,
    private readonly httpOptions: HttpOptions,
  ) {
    this.manifest = manifest
  }

  static async open(manifestUrl: string | URL, options: OpenOptions = {}): Promise<ParqDB> {
    const url = new URL(manifestUrl)
    if (!url.pathname.endsWith('/manifest.json') && !url.pathname.endsWith('manifest.json')) {
      throw new Error('ParqDB.open requires the exact top-level manifest.json URL')
    }
    const manifest = parseManifest(await fetchManifest(url, options))
    if (manifest.postings.files.length + 2 > (options.maxObjects ?? 1_000_000)) {
      throw new Error('package object count exceeds client limit')
    }
    if (manifest.index.dimension > (options.maxDimension ?? 65_536)) {
      throw new Error('package vector dimension exceeds client limit')
    }
    if (manifest.index.nlist > (options.maxNlist ?? 1_000_000)) {
      throw new Error('package nlist exceeds client limit')
    }
    const kernel = await WasmKernel.load(options.wasm)
    return new ParqDB(manifest, url, kernel, options)
  }

  async search(query: Iterable<number>, options: SearchOptions): Promise<SearchHit[]> {
    const { nlist, dimension } = this.manifest.index
    if (!Number.isSafeInteger(options.nprobe) || options.nprobe <= 0) {
      throw new Error('nprobe must be a positive portable integer')
    }
    if (!Number.isSafeInteger(options.k) || options.k <= 0) {
      throw new Error('k must be a positive portable integer')
    }
    const effectiveNprobe = Math.min(options.nprobe, nlist)
    const effectiveK = Math.min(options.k, this.manifest.index.ntotal, 100)
    const rawQuery = Float32Array.from(query)
    if (rawQuery.length !== dimension || rawQuery.some(value => !Number.isFinite(value))) {
      throw new Error(`query must contain ${dimension} finite values`)
    }
    const transformedQuery = this.manifest.index.metric === 'cosine' ? normalize(rawQuery) : rawQuery
    const requestOptions: HttpOptions = { ...this.httpOptions }
    if (options.signal !== undefined) requestOptions.signal = options.signal
    options.trace?.({ phase: 'routing', status: 'start' })
    const selectedCids = await this.selectCids(transformedQuery, effectiveNprobe, requestOptions)
    options.trace?.({ phase: 'routing', status: 'complete', selectedCids: selectedCids.length })
    const selected = new Set(selectedCids)
    const files = this.manifest.postings.files.filter(file => intersects(file, selectedCids))
    options.trace?.({ phase: 'planning', status: 'start', files: files.length })
    const plannedFiles = await Promise.all(
      files.map(async file => ({ file, planned: await this.planPostings(file, selected, requestOptions) })),
    )
    const candidateRows = plannedFiles.reduce(
      (total, entry) => total + entry.planned.spans.reduce((sum, span) => sum + span.end - span.start, 0),
      0,
    )
    if (candidateRows > (options.maxCandidateRows ?? 10_000_000)) {
      throw new Error('selected postings rows exceed client limit')
    }
    options.trace?.({
      phase: 'planning',
      status: 'complete',
      files: files.length,
      candidateRows,
    })
    const candidates: Candidate[] = []
    let order = 0
    let scoringStarted = false
    for (const { file, planned } of plannedFiles) {
      const rangeFile = new HttpRangeBuffer(
        objectUrl(this.manifestUrl, file.path),
        file.size,
        requestOptions,
      )
      for (const span of planned.spans) {
        const rows = await parquetReadObjects({
          file: rangeFile,
          metadata: planned.metadata,
          rowStart: span.start,
          rowEnd: span.end,
          columns: [
            'cid',
            ...this.manifest.index.sourceKeyFields.map((_, position) => `key_${position + 1}`),
            'offset',
            'scale',
            'code',
          ],
          compressors,
          utf8: false,
        })
        const batch = lvqBatch(rows, selected, dimension, this.manifest.index.postingEncoding)
        if (!scoringStarted) {
          scoringStarted = true
          options.trace?.({ phase: 'scoring', status: 'start', candidateRows })
        }
        for (const hit of this.kernel.lvqTopk(
          batch.codes,
          batch.offsets,
          batch.scales,
          transformedQuery,
          this.manifest.index.postingEncoding === 'lvq4' ? 4 : 8,
          effectiveK,
        )) {
          candidates.push({ row: rows[hit.row]!, distance: hit.distance, order })
          order += 1
        }
      }
    }
    options.trace?.({ phase: 'scoring', status: 'complete', candidateRows })
    candidates.sort((left, right) => left.distance - right.distance || left.order - right.order)
    const scale = this.manifest.index.metric === 'cosine' ? 0.5 : 1
    return candidates.slice(0, effectiveK).map(candidate =>
      resultRow(candidate.row, this.manifest.index.sourceKeyFields, candidate.distance * scale),
    )
  }

  private async selectCids(query: Float32Array, nprobe: number, options: HttpOptions): Promise<number[]> {
    const { codes, offsets, scales, cids } = await this.loadCentroids(options)
    return this.kernel.lvqTopk(codes, offsets, scales, query, 8, nprobe).map(hit => cids[hit.row]!)
  }

  private async loadCentroids(options: HttpOptions): Promise<{
    codes: Uint8Array
    offsets: Float32Array
    scales: Float32Array
    cids: Int32Array
  }> {
    if (this.centroidData !== undefined) return this.centroidData
    const loading = this.readCentroids(options)
    this.centroidData = loading
    try {
      return await loading
    } catch (error) {
      if (this.centroidData === loading) this.centroidData = undefined
      throw error
    }
  }

  private async readCentroids(options: HttpOptions): Promise<{
    codes: Uint8Array
    offsets: Float32Array
    scales: Float32Array
    cids: Int32Array
  }> {
    const descriptor = this.manifest.hierarchy.centroids
    const file = new HttpRangeBuffer(objectUrl(this.manifestUrl, descriptor.path), descriptor.size, options)
    const metadata = await parquetMetadataAsync(file)
    if (metadata.num_rows !== BigInt(this.manifest.index.nlist)) {
      throw new Error('leaf centroid count does not match manifest nlist')
    }
    if (metadata.row_groups.length !== this.manifest.hierarchy.rootCount) {
      throw new Error('leaf centroid row groups do not match root-count')
    }
    const rows = await parquetReadObjects({
      file,
      metadata,
      columns: ['cid', 'cid_bucket', 'offset', 'scale', 'code'],
      compressors,
      utf8: false,
    })
    const codes = new Uint8Array(rows.length * this.manifest.index.dimension)
    const offsets = new Float32Array(rows.length)
    const scales = new Float32Array(rows.length)
    const cids = new Int32Array(rows.length)
    rows.forEach((row, position) => {
      const cid = requiredInteger(row.cid, 'centroid cid')
      const bucket = requiredInteger(row.cid_bucket, 'centroid cid_bucket')
      if (
        cid !== position ||
        bucket < 0 ||
        bucket >= this.manifest.hierarchy.rootCount ||
        cid < this.manifest.hierarchy.cidOffsets[bucket]! ||
        cid >= this.manifest.hierarchy.cidOffsets[bucket + 1]!
      ) {
        throw new Error('centroid rows do not follow manifest CID topology')
      }
      const offset = requiredFloat(row.offset, 'centroid LVQ offset')
      const scale = requiredFloat(row.scale, 'centroid LVQ scale')
      if (scale < 0) throw new Error('centroid LVQ scale must be non-negative')
      if (!(row.code instanceof Uint8Array) || row.code.length !== this.manifest.index.dimension) {
        const kind = Object.prototype.toString.call(row.code)
        const length = row.code !== null && typeof row.code === 'object' && 'length' in row.code ? row.code.length : 'missing'
        throw new Error(`centroid LVQ8 code has the wrong shape (${kind}, length ${String(length)})`)
      }
      codes.set(row.code, position * this.manifest.index.dimension)
      offsets[position] = offset
      scales[position] = scale
      cids[position] = cid
    })
    return { codes, offsets, scales, cids }
  }

  private async planPostings(
    file: PostingsFile,
    selected: Set<number>,
    options: HttpOptions,
  ): Promise<{ metadata: FileMetaData; spans: RowSpan[] }> {
    const rangeFile = new HttpRangeBuffer(objectUrl(this.manifestUrl, file.path), file.size, options)
    let loading = this.postingsMetadata.get(file.path)
    if (loading === undefined) {
      loading = parquetMetadataAsync(rangeFile)
      this.postingsMetadata.set(file.path, loading)
    }
    let metadata: FileMetaData
    try {
      metadata = await loading
    } catch (error) {
      if (this.postingsMetadata.get(file.path) === loading) this.postingsMetadata.delete(file.path)
      throw error
    }
    if (metadata.num_rows !== BigInt(file.rows)) throw new Error('postings file row count does not match manifest')
    const spans: RowSpan[] = []
    let rowStart = 0
    for (const rowGroup of metadata.row_groups) {
      const rows = portableBigint(rowGroup.num_rows, 'row-group rows')
      const cid = rowGroupCid(rowGroup)
      if (cid < file.minCid || cid > file.maxCid) throw new Error('row-group CID is outside manifest file range')
      if (selected.has(cid)) appendSpan(spans, rowStart, rowStart + rows)
      rowStart += rows
    }
    if (rowStart !== file.rows) throw new Error('postings row groups do not cover manifest rows')
    return { metadata, spans }
  }
}

function intersects(file: PostingsFile, selected: number[]): boolean {
  return selected.some(cid => cid >= file.minCid && cid <= file.maxCid)
}

function appendSpan(spans: RowSpan[], start: number, end: number): void {
  const previous = spans.at(-1)
  if (previous?.end === start) previous.end = end
  else spans.push({ start, end })
}

function rowGroupCid(rowGroup: RowGroup): number {
  const column = rowGroup.columns.find(entry => entry.meta_data?.path_in_schema.length === 1 && entry.meta_data.path_in_schema[0] === 'cid')
  const statistics = column?.meta_data?.statistics
  if (
    statistics === undefined ||
    statistics.null_count !== 0n ||
    statistics.is_min_value_exact === false ||
    statistics.is_max_value_exact === false
  ) {
    throw new Error('postings row group requires exact, non-null CID statistics')
  }
  const minimum = statistics.min_value ?? statistics.min
  const maximum = statistics.max_value ?? statistics.max
  const minCid = requiredInteger(minimum, 'row-group minimum CID')
  const maxCid = requiredInteger(maximum, 'row-group maximum CID')
  if (minCid !== maxCid) throw new Error('postings row group contains multiple CIDs')
  return minCid
}

function lvqBatch(
  rows: Record<string, unknown>[],
  selected: Set<number>,
  dimension: number,
  encoding: 'lvq4' | 'lvq8',
): { codes: Uint8Array; offsets: Float32Array; scales: Float32Array } {
  const codeSize = encoding === 'lvq4' ? Math.ceil(dimension / 2) : dimension
  const codes = new Uint8Array(rows.length * codeSize)
  const offsets = new Float32Array(rows.length)
  const scales = new Float32Array(rows.length)
  rows.forEach((row, position) => {
    const cid = requiredInteger(row.cid, 'posting cid')
    if (!selected.has(cid)) throw new Error('decoded postings contain an unselected CID')
    if (!(row.code instanceof Uint8Array) || row.code.length !== codeSize) throw new Error('invalid LVQ code')
    const offset = requiredFloat(row.offset, 'LVQ offset')
    const scale = requiredFloat(row.scale, 'LVQ scale')
    if (scale < 0) throw new Error('LVQ scale must be non-negative')
    codes.set(row.code, position * codeSize)
    offsets[position] = offset
    scales[position] = scale
  })
  return { codes, offsets, scales }
}

function resultRow(row: Record<string, unknown>, fields: SourceKeyField[], distance: number): SearchHit {
  const result: Record<string, SourceKeyValue | number> = {}
  fields.forEach((field, position) => {
    const value = row[`key_${position + 1}`]
    validateSourceKey(value, field)
    result[field.name] = value
  })
  result._distance = distance
  return result as SearchHit
}

function validateSourceKey(value: unknown, field: SourceKeyField): asserts value is SourceKeyValue {
  const fixed = /^fixed\(([1-9][0-9]*)\)$/.exec(field.type)
  const valid =
    (field.type === 'boolean' && typeof value === 'boolean') ||
    (field.type === 'int' && typeof value === 'number' && Number.isInteger(value)) ||
    (field.type === 'long' && typeof value === 'bigint') ||
    (field.type === 'string' && typeof value === 'string') ||
    (field.type === 'date' && value instanceof Date) ||
    (field.type === 'binary' && value instanceof Uint8Array) ||
    (fixed !== null && value instanceof Uint8Array && value.length === Number(fixed[1]))
  if (!valid) throw new Error(`source key ${field.name} does not match ${field.type}`)
}

function requiredInteger(value: unknown, label: string): number {
  const converted = typeof value === 'bigint' ? Number(value) : value
  if (!Number.isSafeInteger(converted)) throw new Error(`${label} must be a portable integer`)
  return converted as number
}

function requiredFloat(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${label} must be finite`)
  return value
}

function portableBigint(value: bigint, label: string): number {
  if (value <= 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} is not portable`)
  return Number(value)
}

function normalize(query: Float32Array): Float32Array {
  let squaredNorm = 0
  for (const value of query) squaredNorm += value * value
  if (!Number.isFinite(squaredNorm) || squaredNorm <= 0) throw new Error('cosine query must have a finite, non-zero norm')
  const inverseNorm = 1 / Math.sqrt(squaredNorm)
  return Float32Array.from(query, value => value * inverseNorm)
}

export type { PackageManifest, SourceKeyField, WasmSource }

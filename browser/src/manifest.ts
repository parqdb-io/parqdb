const JSON_SAFE_INTEGER_MAX = 9_007_199_254_740_991
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const SHA256 = /^[0-9a-f]{64}$/

export type Metric = 'l2_squared' | 'cosine'
export type PostingEncoding = 'lvq4' | 'lvq8'

export interface SourceKeyField {
  name: string
  type: string
}

export interface ArtifactObject {
  path: string
  size: number
  sha256: string
}

export interface PostingsFile extends ArtifactObject {
  cidBucket: number
  minCid: number
  maxCid: number
  rows: number
}

export interface SourceFile extends ArtifactObject {
  rowBegin: number
  rowEnd: number
}

export interface PublishedSource {
  rows: number
  rowGroupRows: number
  key: SourceKeyField
  columns: string[]
  files: SourceFile[]
}

export interface EmbeddingDescriptor {
  repository: string
  revision: string
  runtime: 'onnx'
  onnxFile: string
  dimension: number
  maxLength: number
  pooling: string
  normalize: boolean
  inputTemplate: string
  parityProbe: {
    text: string
    vector: number[]
    maxAbsoluteError: number
  }
  assets: ArtifactObject[]
}

export interface IndexArtifactManifest {
  formatVersion: 1
  artifactUuid: string
  index: {
    vectorField: string
    metric: Metric
    postingEncoding: PostingEncoding
    dimension: number
    nlist: number
    ntotal: number
    sourceKeyFields: SourceKeyField[]
  }
  hierarchy: {
    rootCount: number
    cidOffsets: number[]
    centroidEncoding: 'lvq8'
    centroids: ArtifactObject
  }
  postings: {
    files: PostingsFile[]
  }
  source?: PublishedSource
  embedding?: EmbeddingDescriptor
}

export function parseManifest(bytes: ArrayBuffer): IndexArtifactManifest {
  const raw = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  const value = new StrictJsonParser(raw).parse()
  const root = object(value, 'manifest')
  exactKeys(root, [
    'format-version',
    'artifact-uuid',
    'index',
    'hierarchy',
    'postings',
    ...(root.source === undefined ? [] : ['source']),
    ...(root.embedding === undefined ? [] : ['embedding']),
  ])
  integer(root['format-version'], 'format-version', 1, 1)
  const artifactUuid = string(root['artifact-uuid'], 'artifact-uuid')
  if (!UUID.test(artifactUuid) || /^0{8}-0{4}-0{4}-0{4}-0{12}$/.test(artifactUuid)) {
    invalid('artifact-uuid must be a non-nil lowercase UUID')
  }

  const index = object(root.index, 'index')
  exactKeys(index, [
    'vector-field',
    'metric',
    'posting-encoding',
    'dimension',
    'nlist',
    'ntotal',
    'source-key-fields',
  ])
  const vectorField = string(index['vector-field'], 'index.vector-field')
  if (vectorField.length === 0 || vectorField === '_distance') invalid('invalid vector-field')
  const metric = string(index.metric, 'index.metric')
  if (metric !== 'l2_squared' && metric !== 'cosine') invalid('unsupported metric')
  const postingEncoding = string(index['posting-encoding'], 'index.posting-encoding')
  if (postingEncoding !== 'lvq4' && postingEncoding !== 'lvq8') {
    invalid('index artifacts require lvq4 or lvq8')
  }
  const dimension = integer(index.dimension, 'index.dimension', 1)
  const nlist = integer(index.nlist, 'index.nlist', 1)
  const ntotal = integer(index.ntotal, 'index.ntotal', nlist)
  const sourceKeyFields = array(index['source-key-fields'], 'index.source-key-fields').map(
    (entry, position) => {
      const field = object(entry, `index.source-key-fields[${position}]`)
      exactKeys(field, ['name', 'type'])
      const name = string(field.name, 'source-key name')
      const type = string(field.type, 'source-key type')
      if (name.length === 0 || name === '_distance' || !validSourceKeyType(type)) invalid('invalid source-key field')
      return { name, type }
    },
  )
  if (sourceKeyFields.length === 0 || new Set(sourceKeyFields.map(field => field.name)).size !== sourceKeyFields.length) {
    invalid('source-key fields must be non-empty and unique')
  }

  const hierarchy = object(root.hierarchy, 'hierarchy')
  exactKeys(hierarchy, ['cid-offsets', 'centroid-encoding', 'centroids'])
  const cidOffsets = array(hierarchy['cid-offsets'], 'hierarchy.cid-offsets').map(
    (entry, position) => integer(entry, `hierarchy.cid-offsets[${position}]`, 0),
  )
  const rootCount = cidOffsets.length - 1
  if (
    rootCount < 1 ||
    cidOffsets[0] !== 0 ||
    cidOffsets.at(-1) !== nlist ||
    cidOffsets.some((offset, position) => position > 0 && offset <= cidOffsets[position - 1]!)
  ) {
    invalid('cid-offsets must strictly partition [0, nlist)')
  }
  const centroidEncoding = string(hierarchy['centroid-encoding'], 'hierarchy.centroid-encoding')
  if (centroidEncoding !== 'lvq8') invalid('index artifacts require lvq8 leaf centroids')
  const centroids = artifactObject(hierarchy.centroids, 'hierarchy.centroids')

  const postings = object(root.postings, 'postings')
  exactKeys(postings, ['files'])
  const files = array(postings.files, 'postings.files').map((entry, position) => {
    const file = object(entry, `postings.files[${position}]`)
    exactKeys(file, ['path', 'cid-bucket', 'min-cid', 'max-cid', 'rows', 'size', 'sha256'])
    const base = artifactObject(file, `postings.files[${position}]`, [
      'cid-bucket',
      'min-cid',
      'max-cid',
      'rows',
    ])
    const cidBucket = integer(file['cid-bucket'], 'cid-bucket', 0, rootCount - 1)
    const minCid = integer(file['min-cid'], 'min-cid', cidOffsets[cidBucket]!)
    const maxCid = integer(file['max-cid'], 'max-cid', minCid, cidOffsets[cidBucket + 1]! - 1)
    const rows = integer(file.rows, 'rows', 1)
    const prefix = `ivf_postings/cid_bucket=${cidBucket.toString().padStart(6, '0')}/`
    if (!base.path.startsWith(prefix)) invalid('postings path does not match cid-bucket')
    return { ...base, cidBucket, minCid, maxCid, rows }
  })
  if (files.length === 0) invalid('postings inventory must not be empty')
  const allPaths = [centroids.path, ...files.map(file => file.path)]
  if (new Set(allPaths).size !== allPaths.length) invalid('index object paths must be unique')
  for (let position = 1; position < files.length; position += 1) {
    const previous = files[position - 1]!
    const current = files[position]!
    if (compareFile(previous, current) >= 0 || (previous.cidBucket === current.cidBucket && current.minCid < previous.maxCid)) {
      invalid('postings files are not canonically ordered')
    }
  }
  if (files.reduce((sum, file) => sum + file.rows, 0) !== ntotal) {
    invalid('postings rows do not sum to ntotal')
  }

  const source = root.source === undefined
    ? undefined
    : publishedSource(root.source, sourceKeyFields, ntotal)
  const embedding = root.embedding === undefined
    ? undefined
    : embeddingDescriptor(root.embedding, dimension)

  const publicationPaths = [
    centroids.path,
    ...files.map(file => file.path),
    ...(source?.files.map(file => file.path) ?? []),
    ...(embedding?.assets.map(asset => asset.path) ?? []),
  ]
  if (new Set(publicationPaths).size !== publicationPaths.length) {
    invalid('publication object paths must be globally unique')
  }

  return {
    formatVersion: 1,
    artifactUuid,
    index: { vectorField, metric, postingEncoding, dimension, nlist, ntotal, sourceKeyFields },
    hierarchy: { rootCount, cidOffsets, centroidEncoding, centroids },
    postings: { files },
    ...(source === undefined ? {} : { source }),
    ...(embedding === undefined ? {} : { embedding }),
  }
}

function publishedSource(
  value: unknown,
  sourceKeyFields: SourceKeyField[],
  ntotal: number,
): PublishedSource {
  const source = object(value, 'source')
  exactKeys(source, ['rows', 'row-group-rows', 'key', 'columns', 'files'])
  const rows = integer(source.rows, 'source.rows', 1)
  if (rows !== ntotal) invalid('source rows must equal index ntotal')
  const rowGroupRows = integer(source['row-group-rows'], 'source.row-group-rows', 1)
  const rawKey = object(source.key, 'source.key')
  exactKeys(rawKey, ['name', 'type'])
  const key = {
    name: string(rawKey.name, 'source.key.name'),
    type: string(rawKey.type, 'source.key.type'),
  }
  if (
    key.type !== 'long' ||
    sourceKeyFields.length !== 1 ||
    sourceKeyFields[0]!.name !== key.name ||
    sourceKeyFields[0]!.type !== key.type
  ) {
    invalid('source key must match the sole long index source key')
  }
  const columns = array(source.columns, 'source.columns').map((entry, position) =>
    string(entry, `source.columns[${position}]`),
  )
  if (
    columns.length === 0 ||
    columns.some(column => column.length === 0) ||
    new Set(columns).size !== columns.length ||
    !columns.includes(key.name)
  ) {
    invalid('source columns must be non-empty and include its key')
  }
  let expectedBegin = 0
  const files = array(source.files, 'source.files').map((entry, position) => {
    const file = object(entry, `source.files[${position}]`)
    const base = artifactObject(file, `source.files[${position}]`, ['row-begin', 'row-end'])
    const rowBegin = integer(file['row-begin'], 'source file row-begin', 0, rows - 1)
    const rowEnd = integer(file['row-end'], 'source file row-end', rowBegin + 1, rows)
    if (rowBegin !== expectedBegin) invalid('source files must canonically partition its rows')
    expectedBegin = rowEnd
    return { ...base, rowBegin, rowEnd }
  })
  if (files.length === 0 || expectedBegin !== rows) {
    invalid('source files must canonically partition its rows')
  }
  return { rows, rowGroupRows, key, columns, files }
}

function embeddingDescriptor(value: unknown, indexDimension: number): EmbeddingDescriptor {
  const embedding = object(value, 'embedding')
  exactKeys(embedding, [
    'repository', 'revision', 'runtime', 'onnx-file', 'dimension', 'max-length',
    'pooling', 'normalize', 'input-template', 'parity-probe', 'assets',
  ])
  const repository = nonEmptyString(embedding.repository, 'embedding.repository')
  const revision = nonEmptyString(embedding.revision, 'embedding.revision')
  const runtime = string(embedding.runtime, 'embedding.runtime')
  if (runtime !== 'onnx') invalid('embedding runtime must be onnx')
  const onnxFile = string(embedding['onnx-file'], 'embedding.onnx-file')
  validatePath(onnxFile, false)
  const dimension = integer(embedding.dimension, 'embedding.dimension', 1)
  if (dimension !== indexDimension) invalid('embedding dimension must match the index')
  const maxLength = integer(embedding['max-length'], 'embedding.max-length', 1)
  const pooling = nonEmptyString(embedding.pooling, 'embedding.pooling')
  if (typeof embedding.normalize !== 'boolean') invalid('embedding.normalize must be a boolean')
  const normalize = embedding.normalize
  const inputTemplate = nonEmptyString(embedding['input-template'], 'embedding.input-template')
  const probe = object(embedding['parity-probe'], 'embedding.parity-probe')
  exactKeys(probe, ['text', 'vector', 'max-absolute-error'])
  const text = nonEmptyString(probe.text, 'embedding.parity-probe.text')
  const vector = array(probe.vector, 'embedding.parity-probe.vector').map((entry, position) =>
    finiteNumber(entry, `embedding.parity-probe.vector[${position}]`),
  )
  if (vector.length !== dimension) invalid('embedding parity vector has the wrong dimension')
  const maxAbsoluteError = finiteNumber(
    probe['max-absolute-error'],
    'embedding.parity-probe.max-absolute-error',
  )
  if (maxAbsoluteError <= 0) invalid('embedding parity error must be positive')
  const assets = array(embedding.assets, 'embedding.assets').map((entry, position) =>
    artifactObject(entry, `embedding.assets[${position}]`, [], false),
  )
  if (assets.length === 0 || !assets.some(asset => asset.path === onnxFile)) {
    invalid('embedding assets must contain its ONNX file')
  }
  return {
    repository, revision, runtime, onnxFile, dimension, maxLength, pooling,
    normalize, inputTemplate,
    parityProbe: { text, vector, maxAbsoluteError },
    assets,
  }
}

function artifactObject(
  value: unknown,
  label: string,
  extraKeys: string[] = [],
  parquet = true,
): ArtifactObject {
  const entry = object(value, label)
  exactKeys(entry, ['path', 'size', 'sha256', ...extraKeys])
  const path = string(entry.path, `${label}.path`)
  validatePath(path, parquet)
  const size = integer(entry.size, `${label}.size`, 1)
  const sha256 = string(entry.sha256, `${label}.sha256`)
  if (!SHA256.test(sha256)) invalid(`${label}.sha256 is not lowercase SHA-256`)
  return { path, size, sha256 }
}

function compareFile(left: PostingsFile, right: PostingsFile): number {
  return (
    left.cidBucket - right.cidBucket ||
    left.minCid - right.minCid ||
    left.maxCid - right.maxCid ||
    left.path.localeCompare(right.path)
  )
}

function validatePath(path: string, parquet = true): void {
  if (
    path.length === 0 ||
    path.startsWith('/') ||
    path.includes('\\') ||
    path.includes('?') ||
    path.includes('#') ||
    path.includes('://') ||
    path.split('/').some(part => part.length === 0 || part === '.' || part === '..') ||
    (parquet && !path.endsWith('.parquet'))
  ) {
    invalid('invalid artifact object path')
  }
}

function nonEmptyString(value: unknown, label: string): string {
  const result = string(value, label)
  if (result.length === 0) invalid(`${label} must not be empty`)
  return result
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) invalid(`${label} must be finite`)
  return value
}

function validSourceKeyType(type: string): boolean {
  if (['boolean', 'int', 'long', 'binary', 'string', 'date'].includes(type)) return true
  const match = /^fixed\(([1-9][0-9]*)\)$/.exec(type)
  return match !== null && Number(match[1]) <= 0xffff_ffff
}

function exactKeys(value: Record<string, unknown>, expected: string[]): void {
  const expectedSet = new Set(expected)
  if (Object.keys(value).length !== expected.length || Object.keys(value).some(key => !expectedSet.has(key))) {
    invalid('JSON object has unknown or missing fields')
  }
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) invalid(`${label} must be an object`)
  return value as Record<string, unknown>
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) invalid(`${label} must be an array`)
  return value
}

function string(value: unknown, label: string): string {
  if (typeof value !== 'string') invalid(`${label} must be a string`)
  return value
}

function integer(value: unknown, label: string, minimum: number, maximum = JSON_SAFE_INTEGER_MAX): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    invalid(`${label} must be a portable integer in [${minimum}, ${maximum}]`)
  }
  return value as number
}

function invalid(message: string): never {
  throw new Error(`invalid ParqDB artifact manifest: ${message}`)
}

class StrictJsonParser {
  private position = 0

  constructor(private readonly source: string) {}

  parse(): unknown {
    const value = this.value()
    this.space()
    if (this.position !== this.source.length) invalid('trailing JSON input')
    return value
  }

  private value(): unknown {
    this.space()
    const token = this.source[this.position]
    if (token === '{') return this.object()
    if (token === '[') return this.array()
    if (token === '"') return this.string()
    for (const [literal, value] of [['true', true], ['false', false], ['null', null]] as const) {
      if (this.source.startsWith(literal, this.position)) {
        this.position += literal.length
        return value
      }
    }
    const match = /^-?(?:0|[1-9][0-9]*)/.exec(this.source.slice(this.position))
    if (match === null) invalid(`unexpected JSON token at ${this.position}`)
    this.position += match[0].length
    const value = Number(match[0])
    if (!Number.isSafeInteger(value)) invalid('JSON integer is not portable')
    return value
  }

  private object(): Record<string, unknown> {
    this.position += 1
    const result: Record<string, unknown> = {}
    const keys = new Set<string>()
    this.space()
    if (this.consume('}')) return result
    while (true) {
      this.space()
      if (this.source[this.position] !== '"') invalid('JSON object key must be a string')
      const key = this.string()
      if (keys.has(key)) invalid(`duplicate JSON key ${key}`)
      keys.add(key)
      this.space()
      if (!this.consume(':')) invalid('JSON object key requires a value')
      result[key] = this.value()
      this.space()
      if (this.consume('}')) return result
      if (!this.consume(',')) invalid('JSON object entries require a comma')
    }
  }

  private array(): unknown[] {
    this.position += 1
    const result: unknown[] = []
    this.space()
    if (this.consume(']')) return result
    while (true) {
      result.push(this.value())
      this.space()
      if (this.consume(']')) return result
      if (!this.consume(',')) invalid('JSON array entries require a comma')
    }
  }

  private string(): string {
    const start = this.position
    this.position += 1
    while (this.position < this.source.length) {
      const token = this.source[this.position]!
      if (token === '"') {
        this.position += 1
        try {
          return JSON.parse(this.source.slice(start, this.position)) as string
        } catch {
          invalid('invalid JSON string')
        }
      }
      if (token === '\\') this.position += 1
      this.position += 1
    }
    invalid('unterminated JSON string')
  }

  private space(): void {
    while (/\s/.test(this.source[this.position] ?? '')) this.position += 1
  }

  private consume(token: string): boolean {
    if (this.source[this.position] !== token) return false
    this.position += 1
    return true
  }
}

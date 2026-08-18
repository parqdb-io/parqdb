const JSON_SAFE_INTEGER_MAX = 9_007_199_254_740_991
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/
const SHA256 = /^[0-9a-f]{64}$/

export type Metric = 'l2_squared' | 'cosine'
export type PostingEncoding = 'lvq4' | 'lvq8'

export interface SourceKeyField {
  name: string
  type: string
}

export interface PackageObject {
  path: string
  size: number
  sha256: string
}

export interface PostingsFile extends PackageObject {
  cidBucket: number
  minCid: number
  maxCid: number
  rows: number
}

export interface PackageManifest {
  formatVersion: 1
  packageUuid: string
  index: {
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
    roots: PackageObject
    centroids: PackageObject
  }
  postings: {
    files: PostingsFile[]
  }
}

export function parseManifest(bytes: ArrayBuffer): PackageManifest {
  const raw = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  const value = new StrictJsonParser(raw).parse()
  const root = object(value, 'manifest')
  exactKeys(root, ['format-version', 'package-uuid', 'index', 'hierarchy', 'postings'])
  integer(root['format-version'], 'format-version', 1, 1)
  const packageUuid = string(root['package-uuid'], 'package-uuid')
  if (!UUID.test(packageUuid) || /^0{8}-0{4}-0{4}-0{4}-0{12}$/.test(packageUuid)) {
    invalid('package-uuid must be a non-nil lowercase UUID')
  }

  const index = object(root.index, 'index')
  exactKeys(index, [
    'metric',
    'posting-encoding',
    'dimension',
    'nlist',
    'ntotal',
    'source-key-fields',
  ])
  const metric = string(index.metric, 'index.metric')
  if (metric !== 'l2_squared' && metric !== 'cosine') invalid('unsupported metric')
  const postingEncoding = string(index['posting-encoding'], 'index.posting-encoding')
  if (postingEncoding !== 'lvq4' && postingEncoding !== 'lvq8') {
    invalid('static packages require lvq4 or lvq8')
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
  exactKeys(hierarchy, ['root-count', 'cid-offsets', 'roots', 'centroids'])
  const rootCount = integer(hierarchy['root-count'], 'hierarchy.root-count', 1)
  const cidOffsets = array(hierarchy['cid-offsets'], 'hierarchy.cid-offsets').map(
    (entry, position) => integer(entry, `hierarchy.cid-offsets[${position}]`, 0),
  )
  if (
    cidOffsets.length !== rootCount + 1 ||
    cidOffsets[0] !== 0 ||
    cidOffsets.at(-1) !== nlist ||
    cidOffsets.some((offset, position) => position > 0 && offset <= cidOffsets[position - 1]!)
  ) {
    invalid('cid-offsets must strictly partition [0, nlist)')
  }
  const roots = packageObject(hierarchy.roots, 'hierarchy.roots')
  const centroids = packageObject(hierarchy.centroids, 'hierarchy.centroids')
  if (roots.path === centroids.path) invalid('roots and centroids paths must differ')

  const postings = object(root.postings, 'postings')
  exactKeys(postings, ['files'])
  const files = array(postings.files, 'postings.files').map((entry, position) => {
    const file = object(entry, `postings.files[${position}]`)
    exactKeys(file, ['path', 'cid-bucket', 'min-cid', 'max-cid', 'rows', 'size', 'sha256'])
    const base = packageObject(file, `postings.files[${position}]`, [
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
  const allPaths = [roots.path, centroids.path, ...files.map(file => file.path)]
  if (new Set(allPaths).size !== allPaths.length) invalid('package object paths must be unique')
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

  return {
    formatVersion: 1,
    packageUuid,
    index: { metric, postingEncoding, dimension, nlist, ntotal, sourceKeyFields },
    hierarchy: { rootCount, cidOffsets, roots, centroids },
    postings: { files },
  }
}

function packageObject(
  value: unknown,
  label: string,
  extraKeys: string[] = [],
): PackageObject {
  const entry = object(value, label)
  exactKeys(entry, ['path', 'size', 'sha256', ...extraKeys])
  const path = string(entry.path, `${label}.path`)
  validatePath(path)
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

function validatePath(path: string): void {
  if (
    path.length === 0 ||
    path.startsWith('/') ||
    path.includes('\\') ||
    path.includes('?') ||
    path.includes('#') ||
    path.includes('://') ||
    path.split('/').some(part => part.length === 0 || part === '.' || part === '..') ||
    !path.endsWith('.parquet')
  ) {
    invalid('invalid package object path')
  }
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
  throw new Error(`invalid ParqDB package manifest: ${message}`)
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

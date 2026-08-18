import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { resolve } from 'node:path'

import { afterAll, beforeAll, describe, expect, test } from 'vitest'

import { ParqDB } from '../src/index.js'
import { HttpRangeBuffer } from '../src/http.js'
import { parseManifest } from '../src/manifest.js'

const fixtureRoot = resolve('../spec/fixtures/v1/valid/lvq8')
const wasmPath = resolve('../target/wasm32-unknown-unknown/release/parqdb_browser_kernels.wasm')
const requests: Array<{ path: string; range?: string }> = []
const server = createServer(async (request, response) => {
  try {
    const path = new URL(request.url ?? '/', 'http://localhost').pathname.slice(1)
    if (path.includes('..')) throw new Error('invalid path')
    const range = typeof request.headers.range === 'string' ? request.headers.range : undefined
    requests.push(range === undefined ? { path } : { path, range })
    const filePath = resolve(fixtureRoot, path)
    if (!filePath.startsWith(`${fixtureRoot}/`)) throw new Error('path escaped fixture')
    const size = (await stat(filePath)).size
    if (range === undefined) {
      response.writeHead(200, { 'Content-Length': size, 'Accept-Ranges': 'bytes' })
      response.end(await readFile(filePath))
      return
    }
    const match = /^bytes=([0-9]+)-([0-9]+)$/.exec(range)
    if (match === null) throw new Error('invalid range')
    const start = Number(match[1])
    const end = Number(match[2])
    const file = await readFile(filePath)
    response.writeHead(206, {
      'Accept-Ranges': 'bytes',
      'Content-Length': end - start + 1,
      'Content-Range': `bytes ${start}-${end}/${size}`,
    })
    response.end(file.subarray(start, end + 1))
  } catch (error) {
    response.writeHead(404)
    response.end(String(error))
  }
})

let baseUrl = ''

beforeAll(async () => {
  await new Promise<void>(resolveListen => server.listen(0, '127.0.0.1', resolveListen))
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('test server has no TCP address')
  baseUrl = `http://127.0.0.1:${address.port}`
})

afterAll(async () => {
  await new Promise<void>((resolveClose, reject) => {
    server.close(error => (error === undefined ? resolveClose() : reject(error)))
  })
})

describe('static package manifest', () => {
  test('rejects duplicate keys and non-canonical integers', () => {
    const duplicate = new TextEncoder().encode('{"format-version":1,"format-version":1}').buffer
    expect(() => parseManifest(duplicate)).toThrow(/duplicate JSON key/)
    const decimal = new TextEncoder().encode('{"format-version":1.0}').buffer
    expect(() => parseManifest(decimal)).toThrow(/JSON object entries require a comma/)
  })
})

describe('HTTP Range query', () => {
  test('coalesces nearby byte ranges while fetching distant groups concurrently', async () => {
    requests.length = 0
    const path = 'centroids.parquet'
    const bytes = await readFile(resolve(fixtureRoot, path))
    const file = new HttpRangeBuffer(new URL(`${baseUrl}/${path}`), bytes.byteLength, {
      allowHttp: true,
      maxRangeGapBytes: 8,
    })

    const [first, second, third, distant] = await Promise.all([
      file.slice(0, 10),
      file.slice(10, 20),
      file.slice(24, 30),
      file.slice(200, 210),
    ])

    expect(new Uint8Array(first)).toEqual(Uint8Array.from(bytes.subarray(0, 10)))
    expect(new Uint8Array(second)).toEqual(Uint8Array.from(bytes.subarray(10, 20)))
    expect(new Uint8Array(third)).toEqual(Uint8Array.from(bytes.subarray(24, 30)))
    expect(new Uint8Array(distant)).toEqual(Uint8Array.from(bytes.subarray(200, 210)))
    expect(requests.filter(request => request.path === path)).toEqual([
      { path, range: 'bytes=0-29' },
      { path, range: 'bytes=200-209' },
    ])
  })

  test('queries LVQ8 without listing or fetching the native relation manifest', async () => {
    requests.length = 0
    const wasm = await readFile(wasmPath)
    const index = await ParqDB.open(`${baseUrl}/manifest.json`, {
      allowHttp: true,
      wasm,
    })
    const hits = await index.search([0, 0, 0], { nprobe: 1, k: 2 })

    expect(hits.map(hit => hit.document_id)).toEqual(['a', 'b'])
    expect(hits.map(hit => hit._distance)).toEqual([expect.closeTo(1.2519646, 5), 12])
    expect(requests.some(request => request.range !== undefined)).toBe(true)
    expect(requests.some(request => request.path === 'ivf_postings/manifest.json')).toBe(false)
    expect(requests.some(request => request.path.includes('cid_bucket=000000/part-00000.parquet'))).toBe(true)
    expect(requests.filter(request => request.path === 'centroids.parquet')).toEqual([
      { path: 'centroids.parquet' },
    ])
  })

  test('returns all available candidates when k exceeds the index size', async () => {
    const wasm = await readFile(wasmPath)
    const index = await ParqDB.open(`${baseUrl}/manifest.json`, {
      allowHttp: true,
      wasm,
    })

    const hits = await index.search([0, 0, 0], { nprobe: 20, k: 200 })

    expect(hits).toHaveLength(3)
    expect(hits.map(hit => hit.document_id)).toEqual(['a', 'b', 'c'])
  })
})

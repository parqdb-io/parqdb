import { readFile, readdir, stat } from 'node:fs/promises'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import type { Plugin } from 'vite'
import { defineConfig } from 'vitest/config'

const browserRoot = dirname(fileURLToPath(import.meta.url))
const fixtureRoot = resolve(browserRoot, '../spec/fixtures/v1/valid')
const wasmPath = resolve(browserRoot, '../target/wasm32-unknown-unknown/release/parqdb_browser_kernels.wasm')

async function files(root: string): Promise<string[]> {
  const entries = await readdir(root, { withFileTypes: true })
  const nested = await Promise.all(entries.map(entry => {
    const path = join(root, entry.name)
    return entry.isDirectory() ? files(path) : [path]
  }))
  return nested.flat()
}

function staticPackages(): Plugin {
  return {
    name: 'parqdb-static-packages',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        try {
          const pathname = new URL(request.url ?? '/', 'http://localhost').pathname
          let path: string | undefined
          if (pathname === '/parqdb_browser_kernels.wasm') path = wasmPath
          if (pathname.startsWith('/fixtures/')) {
            const relative = pathname.slice('/fixtures/'.length)
            if (!/^(?:lvq4|lvq8)\/[A-Za-z0-9_./=-]+$/.test(relative) || relative.includes('..')) {
              response.writeHead(400).end('invalid fixture path')
              return
            }
            path = resolve(fixtureRoot, relative)
          }
          if (path === undefined) {
            next()
            return
          }
          const size = (await stat(path)).size
          const range = request.headers.range
          const mime = path.endsWith('.json')
            ? 'application/json'
            : path.endsWith('.wasm')
              ? 'application/wasm'
              : 'application/vnd.apache.parquet'
          if (range === undefined) {
            response.writeHead(200, {
              'Accept-Ranges': 'bytes',
              'Content-Length': size,
              'Content-Type': mime,
            })
            response.end(await readFile(path))
            return
          }
          const match = /^bytes=([0-9]+)-([0-9]+)$/.exec(range)
          if (match === null) {
            response.writeHead(416).end('unsupported range')
            return
          }
          const start = Number(match[1])
          const end = Number(match[2])
          if (start < 0 || end < start || end >= size) {
            response.writeHead(416).end('range outside object')
            return
          }
          const file = await readFile(path)
          response.writeHead(206, {
            'Accept-Ranges': 'bytes',
            'Content-Length': end - start + 1,
            'Content-Range': `bytes ${start}-${end}/${size}`,
            'Content-Type': mime,
          })
          response.end(file.subarray(start, end + 1))
        } catch {
          next()
        }
      })
    },
    async generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'parqdb_browser_kernels.wasm',
        source: await readFile(wasmPath),
      })
      for (const encoding of ['lvq4', 'lvq8']) {
        const root = resolve(fixtureRoot, encoding)
        for (const path of await files(root)) {
          this.emitFile({
            type: 'asset',
            fileName: `fixtures/${encoding}/${relative(root, path)}`,
            source: await readFile(path),
          })
        }
      }
    },
  }
}

export default defineConfig({
  root: resolve(browserRoot, 'demo'),
  base: './',
  plugins: [staticPackages()],
  build: {
    outDir: resolve(browserRoot, 'pages-dist'),
    emptyOutDir: true,
  },
  test: {
    root: browserRoot,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
  },
})

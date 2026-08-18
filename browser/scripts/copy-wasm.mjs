import { copyFile, mkdir } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const packageRoot = fileURLToPath(new URL('../', import.meta.url))
const source = fileURLToPath(
  new URL('../../target/wasm32-unknown-unknown/release/parqdb_browser_kernels.wasm', import.meta.url),
)
const destination = fileURLToPath(new URL('../dist/parqdb_browser_kernels.wasm', import.meta.url))

await mkdir(`${packageRoot}/dist`, { recursive: true })
await copyFile(source, destination)

export interface KernelHit {
  row: number
  distance: number
}

interface KernelExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory
  parqdb_alloc(byteLength: number): number
  parqdb_free(pointer: number, byteLength: number): void
  parqdb_dense_topk(
    values: number,
    query: number,
    rows: number,
    dimension: number,
    k: number,
    outputRows: number,
    outputDistances: number,
  ): number
  parqdb_lvq_topk(
    codes: number,
    offsets: number,
    scales: number,
    query: number,
    rows: number,
    dimension: number,
    bits: number,
    k: number,
    outputRows: number,
    outputDistances: number,
  ): number
}

export type WasmSource = BufferSource | WebAssembly.Module | URL

export class WasmKernel {
  private constructor(private readonly exports: KernelExports) {}

  static async load(source?: WasmSource): Promise<WasmKernel> {
    const resolved = source ?? new URL('./parqdb_browser_kernels.wasm', import.meta.url)
    let instance: WebAssembly.Instance
    if (resolved instanceof WebAssembly.Module) {
      instance = await WebAssembly.instantiate(resolved, {})
    } else {
      const bytes = resolved instanceof URL ? await fetch(resolved).then(response => response.arrayBuffer()) : resolved
      const result = await WebAssembly.instantiate(bytes, {})
      instance = result instanceof WebAssembly.Instance ? result : result.instance
    }
    const exports = instance.exports as KernelExports
    if (
      !(exports.memory instanceof WebAssembly.Memory) ||
      typeof exports.parqdb_alloc !== 'function' ||
      typeof exports.parqdb_free !== 'function' ||
      typeof exports.parqdb_dense_topk !== 'function' ||
      typeof exports.parqdb_lvq_topk !== 'function'
    ) {
      throw new Error('invalid ParqDB browser kernel module')
    }
    return new WasmKernel(exports)
  }

  denseTopk(values: Float32Array, query: Float32Array, k: number): KernelHit[] {
    if (query.length === 0 || values.length % query.length !== 0) throw new Error('invalid dense matrix shape')
    const rows = values.length / query.length
    return this.withBuffers(
      [values, query],
      Math.min(rows, k),
      ([valuesPointer, queryPointer], outputRows, outputDistances) =>
        this.exports.parqdb_dense_topk(
          valuesPointer!,
          queryPointer!,
          rows,
          query.length,
          k,
          outputRows,
          outputDistances,
        ),
    )
  }

  lvqTopk(
    codes: Uint8Array,
    offsets: Float32Array,
    scales: Float32Array,
    query: Float32Array,
    bits: 4 | 8,
    k: number,
  ): KernelHit[] {
    if (offsets.length !== scales.length) throw new Error('LVQ offset and scale row counts differ')
    const codeSize = bits === 4 ? Math.ceil(query.length / 2) : query.length
    if (codes.length !== offsets.length * codeSize) throw new Error('LVQ code matrix has the wrong shape')
    return this.withBuffers(
      [codes, offsets, scales, query],
      Math.min(offsets.length, k),
      ([codesPointer, offsetsPointer, scalesPointer, queryPointer], outputRows, outputDistances) =>
        this.exports.parqdb_lvq_topk(
          codesPointer!,
          offsetsPointer!,
          scalesPointer!,
          queryPointer!,
          offsets.length,
          query.length,
          bits,
          k,
          outputRows,
          outputDistances,
        ),
    )
  }

  private withBuffers(
    inputs: ArrayBufferView[],
    outputLength: number,
    invoke: (pointers: number[], outputRows: number, outputDistances: number) => number,
  ): KernelHit[] {
    if (!Number.isSafeInteger(outputLength) || outputLength <= 0) throw new Error('top-k must be positive')
    const allocations: Array<{ pointer: number; byteLength: number }> = []
    try {
      const inputPointers = inputs.map(input => this.allocate(input.byteLength, allocations))
      const outputByteLength = outputLength * 4
      const outputRows = this.allocate(outputByteLength, allocations)
      const outputDistances = this.allocate(outputByteLength, allocations)
      inputs.forEach((input, position) => {
        const bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
        new Uint8Array(this.exports.memory.buffer, inputPointers[position]!, bytes.length).set(bytes)
      })
      const count = invoke(inputPointers, outputRows, outputDistances)
      if (count < 0 || count > outputLength) throw new Error(`ParqDB kernel rejected query (${count})`)
      const rows = new Uint32Array(this.exports.memory.buffer, outputRows, count)
      const distances = new Float32Array(this.exports.memory.buffer, outputDistances, count)
      return Array.from({ length: count }, (_, position) => ({
        row: rows[position]!,
        distance: distances[position]!,
      }))
    } finally {
      for (const allocation of allocations.reverse()) {
        this.exports.parqdb_free(allocation.pointer, allocation.byteLength)
      }
    }
  }

  private allocate(byteLength: number, allocations: Array<{ pointer: number; byteLength: number }>): number {
    const pointer = this.exports.parqdb_alloc(byteLength)
    if (pointer === 0) throw new Error('ParqDB kernel allocation failed')
    allocations.push({ pointer, byteLength })
    return pointer
  }
}

export interface AttnAccumulator {
  layers: number
  heads: number
  rows: number[][][][]
}

export function createAccumulator(layers: number, heads: number): AttnAccumulator {
  return {
    layers, heads,
    rows: Array.from({ length: layers }, () => Array.from({ length: heads }, () => [])),
  }
}

export function addAttentionOutput(
  acc: AttnAccumulator, layer: number, dims: number[], data: Float32Array | number[],
): void {
  const [, heads, qLen, kvLen] = dims
  for (let h = 0; h < heads; h++) {
    const existing = acc.rows[layer][h].length
    for (let q = 0; q < qLen; q++) {
      const rowIndex = qLen === 1 ? kvLen - 1 : q + (kvLen - qLen)
      if (rowIndex < existing) continue  // Approach B resends old rows; keep only new
      const offset = (h * qLen + q) * kvLen
      const row: number[] = []
      for (let c = 0; c <= rowIndex && c < kvLen; c++) row.push(data[offset + c])
      acc.rows[layer][h].push(row)
    }
  }
}

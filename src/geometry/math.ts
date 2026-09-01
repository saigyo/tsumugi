import type { GeometryAsset } from './asset'

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb)
}

const r3 = (v: number) => Math.round(v * 1000) / 1000

export function similarityMatrix(vectors: ArrayLike<number>[]): number[][] {
  const n = vectors.length
  const m: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = r3(cosine(vectors[i], vectors[j]))
      m[i][j] = v
      m[j][i] = v
    }
  }
  return m
}

const BYTE_FALLBACK = /^<0x[0-9A-Fa-f]{2}>$/
const CONTROL = /\p{Cc}/u

export function isRenderableToken(text: string): boolean {
  return text.trim().length > 0 && !BYTE_FALLBACK.test(text) && !CONTROL.test(text)
}

export function renderableNeighbors(asset: GeometryAsset, id: number, n: number): Array<{ id: number; sim: number; text: string }> {
  const out: Array<{ id: number; sim: number; text: string }> = []
  for (const nb of asset.neighbors(id)) {
    const text = asset.text(nb.id)
    if (!isRenderableToken(text)) continue
    out.push({ ...nb, text })
    if (out.length === n) break
  }
  return out
}

export function poolRow(row: ArrayLike<number>, cells: number): number[] {
  const len = row.length
  if (len <= cells) return Array.from(row)
  const out: number[] = []
  for (let c = 0; c < cells; c++) {
    const start = Math.floor((c * len) / cells)
    const end = Math.floor(((c + 1) * len) / cells)
    let sum = 0
    for (let i = start; i < end; i++) sum += row[i]
    out.push(sum / (end - start))
  }
  return out
}

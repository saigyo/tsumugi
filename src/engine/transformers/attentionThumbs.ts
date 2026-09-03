import type { AttentionGridCell } from '../../trace/types'
import type { AttnAccumulator } from './attentionAccum'
import type { HeadStats } from './attentionStats'

// Mean-pooled block average of a ragged causal matrix. Mean pooling
// preserves relative mass — max pooling would make every head look like
// its brightest cell. Pooled rows do NOT sum to 1; that is by design.
export function poolThumb(matrix: number[][], buckets = 12): number[][] {
  const n = matrix.length
  if (n === 0) return []
  const b = Math.min(buckets, n)
  const edge = (i: number) => Math.floor((i * n) / b)
  const out: number[][] = []
  for (let br = 0; br < b; br++) {
    const row: number[] = []
    for (let bc = 0; bc < b; bc++) {
      let sum = 0, count = 0
      for (let r = edge(br); r < edge(br + 1); r++) {
        for (let c = edge(bc); c < edge(bc + 1); c++) {
          if (c < matrix[r].length) { sum += matrix[r][c]; count++ }
        }
      }
      row.push(count ? sum / count : 0)
    }
    out.push(row)
  }
  return out
}

export function buildGridCells(acc: AttnAccumulator, stats: HeadStats[]): AttentionGridCell[] {
  return stats.map((s) => ({
    layer: s.layer,
    head: s.head,
    thumb: poolThumb(acc.rows[s.layer][s.head]),
    prevTokenScore: s.prevTokenScore,
    sinkScore: s.sinkScore,
    inductionScore: s.inductionScore,
    corefScore: s.corefScore,
    distinctiveScore: s.distinctiveScore,
  }))
}

import type { AttentionHead, AttentionLabel, TokenInfo } from '../../trace/types'
import type { AttnAccumulator } from './attentionAccum'

export interface HeadStats {
  layer: number
  head: number
  prevTokenScore: number
  sinkScore: number
  inductionScore: number | null
}

export function headStats(acc: AttnAccumulator, tokens: TokenInfo[]): HeadStats[] {
  // induction targets: for row i whose token appeared at j < i, target j+1
  const targets: Array<number | null> = tokens.map((t, i) => {
    for (let j = i - 1; j >= 0; j--) {
      if (tokens[j].text.trim() === t.text.trim() && j + 1 <= i) return j + 1
    }
    return null
  })
  const out: HeadStats[] = []
  for (let l = 0; l < acc.layers; l++) {
    for (let h = 0; h < acc.heads; h++) {
      const m = acc.rows[l][h]
      let prev = 0, sink = 0, n = 0, ind = 0, indN = 0
      for (let i = 1; i < m.length; i++) {
        prev += m[i][i - 1]
        sink += m[i][0]
        n++
        const t = targets[i]
        if (t !== null && t < m[i].length) { ind += m[i][t]; indN++ }
      }
      out.push({
        layer: l, head: h,
        prevTokenScore: n ? prev / n : 0,
        sinkScore: n ? sink / n : 0,
        inductionScore: indN ? ind / indN : null,
      })
    }
  }
  return out
}

const round = (x: number) => Math.round(x * 100) / 100

export function selectShowcaseHeads(
  stats: HeadStats[], acc: AttnAccumulator, threshold = 0.3,
): AttentionHead[] {
  const pick = (label: AttentionLabel, score: (s: HeadStats) => number | null): AttentionHead | null => {
    let best: HeadStats | null = null
    let bestScore = threshold
    for (const s of stats) {
      const v = score(s)
      if (v !== null && v > bestScore) { best = s; bestScore = v }
    }
    if (!best) return null
    return { layer: best.layer, head: best.head, label,
      score: round(bestScore), matrix: acc.rows[best.layer][best.head] }
  }
  return [
    pick('previous-token', (s) => s.prevTokenScore),
    pick('attention-sink', (s) => s.sinkScore),
    pick('induction', (s) => s.inductionScore),
  ].filter((h): h is AttentionHead => h !== null)
}

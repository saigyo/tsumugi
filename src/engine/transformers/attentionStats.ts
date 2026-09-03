import type { AttentionHead, AttentionLabel, TokenInfo } from '../../trace/types'
import type { AttnAccumulator } from './attentionAccum'

export interface HeadStats {
  layer: number
  head: number
  prevTokenScore: number
  sinkScore: number
  inductionScore: number | null
  // mean over pronoun rows of the peak weight on an earlier word token
  // (sink, previous-token and self columns excluded); null without pronoun rows.
  // Antecedent-blind: see docs/research/2026-09-03-coreference-head-spike.md
  corefScore: number | null
  // (1 − best template score) × (1 − mean normalized row entropy):
  // high = peaked attention that matches no known template
  distinctiveScore: number
}

const PRONOUNS = new Set(['it', 'he', 'she', 'they', 'him', 'her', 'his', 'its', 'their', 'them'])
const isPronoun = (t: TokenInfo) => PRONOUNS.has(t.text.trim().toLowerCase())
// Function words are never referents. Without this list, heads that park every
// row on a fixed preposition or article (" on", " that") outscore the real
// coreference head, whose target is a noun — seen live on the cat example.
const FUNCTION_WORDS = new Set((
  'a an the this that these those and or but nor so yet for of to in on at by with from ' +
  'into onto over under about after before because until while as than then when where if ' +
  'though although whether is are was were be been being am do does did has have had having ' +
  'not no can could may might must shall should will would'
).split(' '))
// a plausible referent column: a content word of at least two letters
const isWordToken = (t: TokenInfo) => {
  const s = t.text.trim()
  return /^\p{L}{2,}$/u.test(s) && !FUNCTION_WORDS.has(s.toLowerCase())
}

export function headStats(acc: AttnAccumulator, tokens: TokenInfo[]): HeadStats[] {
  // induction targets: for row i whose token appeared at j < i, target j+1
  const targets: Array<number | null> = tokens.map((t, i) => {
    if (t.text.trim() === '') return null
    for (let j = i - 1; j >= 0; j--) {
      if (tokens[j].text.trim() !== '' && tokens[j].text.trim() === t.text.trim()) return j + 1
    }
    return null
  })
  // coreference rows: pronouns with at least one candidate column (i ≥ 2)
  const pronounRows = tokens.map((t, i) => i >= 2 && isPronoun(t))
  const wordCols = tokens.map(isWordToken)
  const out: HeadStats[] = []
  for (let l = 0; l < acc.layers; l++) {
    for (let h = 0; h < acc.heads; h++) {
      const m = acc.rows[l][h]
      let prev = 0, sink = 0, n = 0, ind = 0, indN = 0, ent = 0, coref = 0, corefN = 0
      for (let i = 1; i < m.length; i++) {
        prev += m[i][i - 1]
        sink += m[i][0]
        n++
        let rowEnt = 0
        for (const w of m[i]) if (w > 0) rowEnt -= w * Math.log(w)
        ent += Math.min(1, rowEnt / Math.log(m[i].length))
        const t = targets[i]
        if (t !== null && t < m[i].length) { ind += m[i][t]; indN++ }
        if (pronounRows[i]) {
          let peak = 0
          for (let j = 1; j < i - 1 && j < m[i].length; j++) {
            if (wordCols[j] && m[i][j] > peak) peak = m[i][j]
          }
          coref += peak
          corefN++
        }
      }
      const prevTokenScore = n ? prev / n : 0
      const sinkScore = n ? sink / n : 0
      const inductionScore = indN ? ind / indN : null
      const corefScore = corefN ? coref / corefN : null
      const templateMax = Math.max(prevTokenScore, sinkScore, inductionScore ?? 0, corefScore ?? 0)
      const uniformity = n ? ent / n : 1
      out.push({
        layer: l, head: h, prevTokenScore, sinkScore, inductionScore, corefScore,
        distinctiveScore: n ? (1 - templateMax) * (1 - uniformity) : 0,
      })
    }
  }
  return out
}

const round = (x: number) => Math.round(x * 100) / 100

export type ShowcasePrev = Partial<Record<AttentionLabel, { layer: number; head: number }>>

export function selectShowcaseHeads(
  stats: HeadStats[], acc: AttnAccumulator, threshold = 0.3, prev?: ShowcasePrev,
): AttentionHead[] {
  const pick = (
    label: AttentionLabel, score: (s: HeadStats) => number | null, thr = threshold,
  ): AttentionHead | null => {
    let best: HeadStats | null = null
    let bestScore = thr
    for (const s of stats) {
      const v = score(s)
      if (v !== null && v > bestScore) { best = s; bestScore = v }
    }
    // hysteresis: keep last cycle's head unless the challenger beats its
    // CURRENT score by ≥ 0.05; an incumbent fallen to ≤ thr loses its seat
    const p = prev?.[label]
    if (p) {
      const inc = stats.find((s) => s.layer === p.layer && s.head === p.head)
      const incScore = inc ? score(inc) : null
      if (inc && incScore !== null && incScore > thr && (!best || bestScore - incScore < 0.05)) {
        best = inc
        bestScore = incScore
      }
    }
    if (!best) return null
    return { layer: best.layer, head: best.head, label,
      score: round(bestScore), matrix: acc.rows[best.layer][best.head] }
  }
  return [
    pick('previous-token', (s) => s.prevTokenScore),
    pick('attention-sink', (s) => s.sinkScore),
    pick('induction', (s) => s.inductionScore),
    pick('coreference', (s) => s.corefScore),
    pick('distinctive', (s) => s.distinctiveScore, 0.25),
  ].filter((h): h is AttentionHead => h !== null)
}

// Label a single head for the pin flow: best TEMPLATE score at/above the
// threshold ('distinctive' is not a template and is never assigned here).
export function resolveHeadLabel(
  stats: HeadStats[], layer: number, head: number, threshold = 0.3,
): { label: AttentionLabel | null; score: number | null } {
  const s = stats.find((x) => x.layer === layer && x.head === head)
  if (!s) return { label: null, score: null }
  const candidates: Array<[AttentionLabel, number | null]> = [
    ['previous-token', s.prevTokenScore],
    ['attention-sink', s.sinkScore],
    ['induction', s.inductionScore],
    ['coreference', s.corefScore],
  ]
  let label: AttentionLabel | null = null
  let best = threshold
  for (const [lab, v] of candidates) {
    if (v !== null && v >= best) { label = lab; best = v }
  }
  return label ? { label, score: round(best) } : { label: null, score: null }
}

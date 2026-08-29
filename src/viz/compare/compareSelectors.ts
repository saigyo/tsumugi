import type { AttentionHead, TokenInfo, TraceEvent } from '../../trace/types'
import { distributionFor } from '../selectors'

export interface AlignedRuns {
  promptA: TokenInfo[]
  promptB: TokenInfo[]
  chosenA: TokenInfo[]
  chosenB: TokenInfo[]
  maxCycles: number
  samePrompt: boolean
  forkCycle: number | null
}

const promptOf = (events: TraceEvent[]): TokenInfo[] => {
  for (const e of events) if (e.type === 'tokenize') return e.tokens
  return []
}
const chosenOf = (events: TraceEvent[]): TokenInfo[] => {
  const out: TokenInfo[] = []
  for (const e of events) if (e.type === 'append') out[e.cycle] = e.token
  return out
}

export function alignRuns(a: TraceEvent[], b: TraceEvent[]): AlignedRuns {
  const promptA = promptOf(a), promptB = promptOf(b)
  const chosenA = chosenOf(a), chosenB = chosenOf(b)
  const samePrompt = promptA.length === promptB.length && promptA.every((t, i) => t.id === promptB[i].id)
  let forkCycle: number | null = null
  if (samePrompt) {
    for (let c = 0; c < Math.min(chosenA.length, chosenB.length); c++) {
      if (chosenA[c].id !== chosenB[c].id) { forkCycle = c; break }
    }
  }
  return { promptA, promptB, chosenA, chosenB,
    maxCycles: Math.max(chosenA.length, chosenB.length), samePrompt, forkCycle }
}

export interface PairedHead { layer: number; head: number; a?: AttentionHead; b?: AttentionHead }

const headsAt = (events: TraceEvent[], cycle: number): AttentionHead[] => {
  for (const e of events) if (e.type === 'attention' && e.cycle === cycle) return e.heads
  return []
}

export function pairedHeads(a: TraceEvent[], b: TraceEvent[], cycle: number): PairedHead[] {
  const byKey = new Map<string, PairedHead>()
  for (const h of headsAt(a, cycle)) byKey.set(`${h.layer}-${h.head}`, { layer: h.layer, head: h.head, a: h })
  for (const h of headsAt(b, cycle)) {
    const key = `${h.layer}-${h.head}`
    const existing = byKey.get(key)
    if (existing) existing.b = h
    else byKey.set(key, { layer: h.layer, head: h.head, b: h })
  }
  return [...byKey.values()].sort((x, y) => x.layer - y.layer || x.head - y.head)
}

export function pairedDistributions(a: TraceEvent[], b: TraceEvent[], cycle: number) {
  return { a: distributionFor(a, cycle), b: distributionFor(b, cycle) }
}

export interface CombinedRow {
  id: number
  text: string
  pA: number | null   // null = outside run A's stored top-k (counts as 0 in delta)
  pB: number | null
  delta: number       // (pA ?? 0) − (pB ?? 0); a lower bound when approx
  approx: boolean
}

// Token-aligned union of both runs' top-k at one cycle, for the combined
// distributions view. Undefined when either side lacks the cycle — the
// caller falls back to the per-side rendering (which explains run ends).
export function combinedDistribution(a: TraceEvent[], b: TraceEvent[], cycle: number):
  | { rows: CombinedRow[]; chosenA: number; chosenB: number }
  | undefined {
  const dA = distributionFor(a, cycle)
  const dB = distributionFor(b, cycle)
  if (!dA || !dB) return undefined
  const byId = new Map<number, CombinedRow>()
  for (const t of dA.softmax.topK) byId.set(t.id, { id: t.id, text: t.text, pA: t.prob, pB: null, delta: 0, approx: false })
  for (const t of dB.softmax.topK) {
    const row = byId.get(t.id)
    if (row) row.pB = t.prob
    else byId.set(t.id, { id: t.id, text: t.text, pA: null, pB: t.prob, delta: 0, approx: false })
  }
  for (const row of byId.values()) {
    row.approx = row.pA === null || row.pB === null
    row.delta = (row.pA ?? 0) - (row.pB ?? 0)
  }
  return { rows: [...byId.values()], chosenA: dA.sample.chosen.id, chosenB: dB.sample.chosen.id }
}

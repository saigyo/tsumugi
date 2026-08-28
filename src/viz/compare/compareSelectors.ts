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

import type { TokenInfo, TraceEvent } from '../trace/types'

export type StageId = 'tokenizer' | 'embeddings' | 'layers' | 'logits' | 'sampler' | null

const STAGE_OF: Partial<Record<TraceEvent['type'], StageId>> = {
  tokenize: 'tokenizer', embed: 'embeddings', layer: 'layers', attention: 'layers',
  logits: 'logits', softmax: 'logits', sample: 'sampler', append: 'sampler',
}

export function activeStage(e: TraceEvent | undefined): StageId {
  return e ? STAGE_OF[e.type] ?? null : null
}

export function eventAt(events: TraceEvent[], cursor: number): TraceEvent | undefined {
  return cursor >= 0 ? events[cursor] : undefined
}

export function visibleTokens(events: TraceEvent[], cursor: number) {
  const prompt: TokenInfo[] = []
  const generated: Array<TokenInfo & { cycle: number }> = []
  for (const e of events.slice(0, cursor + 1)) {
    if (e.type === 'tokenize') prompt.push(...e.tokens)
    if (e.type === 'append') generated.push({ ...e.token, cycle: e.cycle })
  }
  return { prompt, generated }
}

export function distributionFor(events: TraceEvent[], cycle: number):
  | { softmax: Extract<TraceEvent, { type: 'softmax' }>; sample: Extract<TraceEvent, { type: 'sample' }> }
  | undefined {
  let softmax, sample
  for (const e of events) {
    if (e.type === 'softmax' && e.cycle === cycle) softmax = e
    if (e.type === 'sample' && e.cycle === cycle) sample = e
  }
  return softmax && sample ? { softmax, sample } : undefined
}

export function latestOfType<K extends TraceEvent['type']>(
  events: TraceEvent[], cursor: number, type: K,
): Extract<TraceEvent, { type: K }> | undefined {
  for (let i = Math.min(cursor, events.length - 1); i >= 0; i--)
    if (events[i].type === type) return events[i] as Extract<TraceEvent, { type: K }>
  return undefined
}

export function cycleTickIndices(events: TraceEvent[]): number[] {
  return events.flatMap((e, i) => (e.type === 'append' ? [i] : []))
}

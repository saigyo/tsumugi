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

// Shapes of what flows along each pipeline connector at the current cursor.
export interface FlowShapes {
  ids?: string      // Tokenizer → Embeddings
  stream?: string   // Embeddings → Layers: the residual stream [seq × d_model]
  lastRow?: string  // Layers → Logits: only the last position is read out
  vocab?: string    // Logits → Sampler: one score per vocabulary entry
  loop?: string     // Sampler → token stream
}

const thousands = (n: number) => n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ')

export function flowShapes(events: TraceEvent[], cursor: number): FlowShapes {
  const shapes: FlowShapes = {}
  const tokenize = latestOfType(events, cursor, 'tokenize')
  if (!tokenize) return shapes
  shapes.ids = `[${tokenize.tokens.length}]`
  shapes.loop = '+1 token'
  const embed = latestOfType(events, cursor, 'embed')
  if (embed) {
    shapes.stream = `[${embed.seqLen}×${embed.dims}]`
    shapes.lastRow = `[1×${embed.dims}]`
  }
  const runStart = latestOfType(events, cursor, 'run-start')
  if (runStart?.vocabSize) shapes.vocab = `[${thousands(runStart.vocabSize)}]`
  return shapes
}

// The one event per cycle where a stage's payload is fully on screen.
const REPRESENTATIVE: Record<Exclude<StageId, null>, TraceEvent['type']> = {
  tokenizer: 'tokenize', embeddings: 'embed', layers: 'attention',
  logits: 'softmax', sampler: 'sample',
}

function cycleAt(events: TraceEvent[], cursor: number): number {
  for (let i = Math.min(cursor, events.length - 1); i >= 0; i--) {
    const e = events[i]
    if ('cycle' in e) return e.cycle
  }
  return 0
}

export function stageEventIndex(events: TraceEvent[], cursor: number, stage: Exclude<StageId, null>): number {
  const type = REPRESENTATIVE[stage]
  if (type === 'tokenize') return events.findIndex((e) => e.type === 'tokenize')
  const cycle = cycleAt(events, cursor)
  let fallback = -1
  for (let i = 0; i < events.length; i++) {
    const e = events[i]
    if (e.type !== type) continue
    if ('cycle' in e && e.cycle === cycle) return i
    fallback = i
  }
  return fallback
}

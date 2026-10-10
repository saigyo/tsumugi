import type { EmbedSource, LayerVotes, LogitCandidate, TokenInfo, TraceEvent } from '../trace/types'

export type StageId = 'tokenizer' | 'embeddings' | 'layers' | 'logits' | 'softmax' | 'sampler' | null

const STAGE_OF: Partial<Record<TraceEvent['type'], StageId>> = {
  tokenize: 'tokenizer', embed: 'embeddings', layer: 'layers', attention: 'layers', edits: 'layers',
  logits: 'logits', softmax: 'softmax', sample: 'sampler', append: 'sampler',
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
  vocab?: string    // Logits → Softmax: one score per vocabulary entry
  probs?: string    // Softmax → Sampler: only the top-k survive the cut
  loop?: string     // Sampler → token stream
}

export const thousands = (n: number) => n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ')

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
  if (runStart) shapes.probs = `[${runStart.params.topK}]`
  return shapes
}

// The one event per cycle where a stage's payload is fully on screen.
const REPRESENTATIVE: Record<Exclude<StageId, null>, TraceEvent['type']> = {
  tokenizer: 'tokenize', embeddings: 'embed', layers: 'attention',
  logits: 'logits', softmax: 'softmax', sampler: 'sample',
}

export function viewedCycle(events: TraceEvent[], cursor: number): number {
  for (let i = Math.min(cursor, events.length - 1); i >= 0; i--) {
    const e = events[i]
    if ('cycle' in e) return e.cycle
  }
  return 0
}

export function stageEventIndex(events: TraceEvent[], cursor: number, stage: Exclude<StageId, null>): number {
  const type = REPRESENTATIVE[stage]
  if (type === 'tokenize') return events.findIndex((e) => e.type === 'tokenize')
  const cycle = viewedCycle(events, cursor)
  let fallback = -1
  for (let i = 0; i < events.length; i++) {
    const e = events[i]
    if (e.type !== type) continue
    if ('cycle' in e && e.cycle === cycle) return i
    fallback = i
  }
  return fallback
}

export interface EmbeddingRows {
  tokens: TokenInfo[]
  rows?: number[][]     // one per token, only when every visible position has a model row
  source: EmbedSource
}

// The visible sequence and, when the run recorded exact rows for all of it,
// those rows in position order. Cycle c's embed event carries the rows fed
// that cycle (the prompt at cycle 0, the previous chosen token after), so at
// an embed cursor the rows cover exactly the visible tokens. Any asset-source
// cycle, or a position without a row yet, degrades the whole run to 'asset'
// so the card never mixes exact and reduced vectors.
export function embeddingRows(events: TraceEvent[], cursor: number): EmbeddingRows {
  const { prompt, generated } = visibleTokens(events, cursor)
  const tokens: TokenInfo[] = [...prompt, ...generated.map(({ id, text }) => ({ id, text }))]
  const rows: number[][] = []
  let model = true
  for (const e of events.slice(0, cursor + 1)) {
    if (e.type !== 'embed') continue
    if (e.source !== 'model' || !e.rows) { model = false; break }
    rows.push(...e.rows)
  }
  if (!model || rows.length < tokens.length) return { tokens, source: 'asset' }
  return { tokens, rows: rows.slice(0, tokens.length), source: 'model' }
}

export interface EditRows { tokens: TokenInfo[]; attn: number[][]; mlp: number[][] }

// The edit-size heatmap for the viewed cycle: rows from every edits event up
// to and including that cycle (looked up in the whole trace, like the votes,
// so the Layers stage shows the cycle's rows before the player reaches the
// edits event), cut to the visible tokens. Undefined when the run has none.
export function editRows(events: TraceEvent[], cursor: number): EditRows | undefined {
  const cycle = viewedCycle(events, cursor)
  const attn: number[][] = [], mlp: number[][] = []
  for (const e of events) {
    if (e.type !== 'edits' || e.cycle > cycle) continue
    e.attn.forEach((row, i) => { attn[e.start + i] = row; mlp[e.start + i] = e.mlp[i] })
  }
  let n = 0
  while (n < attn.length && attn[n] && mlp[n]) n++   // contiguous from position 0
  if (n === 0) return undefined
  const { prompt, generated } = visibleTokens(events, cursor)
  const tokens = [...prompt, ...generated.map(({ id, text }) => ({ id, text }))]
  const rows = Math.min(n, tokens.length)
  return { tokens: tokens.slice(0, rows), attn: attn.slice(0, rows), mlp: mlp.slice(0, rows) }
}

export type VotedCandidate = LogitCandidate & { votes: LayerVotes }
export type CycleVotes =
  | { status: 'ok'; candidates: VotedCandidate[] }
  | { status: 'pending' }   // the viewed cycle's logits are not in the trace yet, and the run goes on
  | { status: 'none' }      // they are, without votes for every candidate

// The votes card reads the viewed cycle's logits event, which comes after
// the layers in the trace: on the Layers stage that is a look ahead.
export function cycleVotes(events: TraceEvent[], cursor: number): CycleVotes {
  const cycle = viewedCycle(events, cursor)
  const logits = events.find((e): e is Extract<TraceEvent, { type: 'logits' }> => e.type === 'logits' && e.cycle === cycle)
  if (!logits) return events.some((e) => e.type === 'run-end') ? { status: 'none' } : { status: 'pending' }
  if (!logits.topK.every((c) => c.votes)) return { status: 'none' }
  return { status: 'ok', candidates: logits.topK as VotedCandidate[] }
}

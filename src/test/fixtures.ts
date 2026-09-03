import type { AttentionGridCell, Mode, RunEndReason, TraceEvent, TokenInfo } from '../trace/types'
import type { RunRecord } from '../app/runsStore'

export interface FixtureTraceOpts {
  cycles?: number
  layers?: number
  prompt?: string
  promptTokens?: TokenInfo[]
  // one chosen token per cycle; defaults reproduce makeFixtureTrace exactly
  chosen?: TokenInfo[]
  temperature?: number
  mode?: Mode
  reason?: RunEndReason
  embedRows?: boolean
}

const DEFAULT_WORDS = [' sat', ' on', ' the', ' mat']

// Deterministic 576-dim "embedding" for a token id, for model-source fixtures.
export function fixtureEmbedding(id: number): number[] {
  return Array.from({ length: 576 }, (_, d) => Math.round(Math.sin(id * 0.37 + d * 0.11) * 1000) / 1000)
}

export function buildFixtureTrace(opts: FixtureTraceOpts = {}): TraceEvent[] {
  const cycles = opts.cycles ?? 2
  const layers = opts.layers ?? 3
  const promptTokens = opts.promptTokens ?? [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }]
  const temperature = opts.temperature ?? 0.8
  const chosenFor = (c: number): TokenInfo =>
    opts.chosen?.[c] ?? { id: 100 + c, text: DEFAULT_WORDS[c % DEFAULT_WORDS.length] }
  const events: TraceEvent[] = [
    { type: 'run-start', prompt: opts.prompt ?? 'The cat', mode: opts.mode ?? 'sim', modelId: 'fixture',
      params: { temperature, topK: 10, maxNewTokens: cycles }, vocabSize: 49152 },
    { type: 'tokenize', tokens: promptTokens },
  ]
  for (let c = 0; c < cycles; c++) {
    const chosen = chosenFor(c)
    // the rows fed this cycle: the whole prompt at cycle 0, then the token chosen last cycle
    const fed = c === 0 ? promptTokens.map((t) => t.id) : [chosenFor(c - 1).id]
    events.push(opts.embedRows
      ? { type: 'embed', cycle: c, seqLen: promptTokens.length + c, dims: 576, source: 'model', rows: fed.map(fixtureEmbedding) }
      : { type: 'embed', cycle: c, seqLen: promptTokens.length + c, dims: 576, source: 'asset' })
    for (let l = 0; l < layers; l++) events.push({ type: 'layer', cycle: c, index: l, total: layers })
    const seq = promptTokens.length + c
    const row = (i: number, weights: Array<[number, number]>): number[] => {
      const w = Array.from({ length: i + 1 }, () => 0)
      for (const [pos, mass] of weights) w[Math.min(pos, i)] += mass
      return w
    }
    events.push({ type: 'attention', cycle: c, heads: [
      { layer: 0, head: 3, label: 'attention-sink',
        matrix: Array.from({ length: seq }, (_, i) => i === 0 ? [1] : row(i, [[0, 0.7], [i - 1, 0.2], [i, 0.1]])) },
      { layer: 2, head: 1, label: 'previous-token',
        matrix: Array.from({ length: seq }, (_, i) => i === 0 ? [1] : row(i, [[i - 1, 0.8], [i, 0.2]])) },
    ] })
    events.push({ type: 'logits', cycle: c, topK: [
      { ...chosen, logit: 9.1 }, { id: 200, text: ' ran', logit: 7.2 }, { id: 201, text: ' was', logit: 5.0 },
    ] })
    events.push({ type: 'softmax', cycle: c, temperature, topK: [
      { ...chosen, prob: 0.7 }, { id: 200, text: ' ran', prob: 0.2 }, { id: 201, text: ' was', prob: 0.1 },
    ] })
    events.push({ type: 'sample', cycle: c, chosen, method: 'top-k' })
    events.push({ type: 'append', cycle: c, token: chosen })
  }
  events.push({ type: 'run-end', reason: opts.reason ?? 'max-tokens' })
  return events
}

export function makeFixtureTrace(cycles = 2, layers = 3): TraceEvent[] {
  return buildFixtureTrace({ cycles, layers })
}

export function makeRunRecord(
  seq: number, opts: FixtureTraceOpts & { pinned?: boolean; endedAt?: number; id?: string } = {},
): RunRecord {
  return {
    id: opts.id ?? `run-${seq}`,
    meta: {
      seq,
      prompt: opts.prompt ?? 'The cat',
      params: { temperature: opts.temperature ?? 0.8, topK: 10, maxNewTokens: opts.cycles ?? 2 },
      mode: opts.mode ?? 'sim',
      endedAt: opts.endedAt ?? 1000 + seq,
      reason: opts.reason ?? 'max-tokens',
      pinned: opts.pinned ?? false,
    },
    events: buildFixtureTrace(opts),
  }
}

// Grid tests use this instead of growing makeFixtureTrace — the default
// fixture's event indices are load-bearing for index-based tests.
export function makeGridEvent(layers = 2, heads = 2): Extract<TraceEvent, { type: 'attention-grid' }> {
  const thumb = (l: number, h: number): number[][] =>
    Array.from({ length: 4 }, (_, r) => Array.from({ length: 4 }, (_, c) =>
      c > r ? 0 : ((l * heads + h + 1) / (layers * heads + 1)) * (c === r ? 1 : 0.25)))
  const cells: AttentionGridCell[] = []
  for (let l = 0; l < layers; l++) {
    for (let h = 0; h < heads; h++) {
      const k = l * heads + h
      cells.push({
        layer: l, head: h, thumb: thumb(l, h),
        prevTokenScore: (k % 5) / 5,
        sinkScore: ((k + 1) % 5) / 5,
        inductionScore: k % 3 === 0 ? null : (k % 4) / 4,
        corefScore: k % 2 === 0 ? null : (k % 3) / 3,
        distinctiveScore: ((k * 3 + 1) % 5) / 5,
      })
    }
  }
  return { type: 'attention-grid', layers, heads, cells }
}

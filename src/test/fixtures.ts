import type { AttentionGridCell, TraceEvent, TokenInfo } from '../trace/types'

export function makeFixtureTrace(cycles = 2, layers = 3): TraceEvent[] {
  const events: TraceEvent[] = [
    { type: 'run-start', prompt: 'The cat', mode: 'sim', modelId: 'fixture',
      params: { temperature: 0.8, topK: 10, maxNewTokens: cycles }, vocabSize: 49152 },
    { type: 'tokenize', tokens: [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }] },
  ]
  const words = [' sat', ' on', ' the', ' mat']
  for (let c = 0; c < cycles; c++) {
    const chosen: TokenInfo = { id: 100 + c, text: words[c % words.length] }
    events.push({ type: 'embed', cycle: c, seqLen: 2 + c, dims: 576,
      preview: [[0.1, -0.2, 0.3], [0.0, 0.5, -0.1]] })
    for (let l = 0; l < layers; l++) events.push({ type: 'layer', cycle: c, index: l, total: layers })
    const seq = 2 + c
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
    events.push({ type: 'softmax', cycle: c, temperature: 0.8, topK: [
      { ...chosen, prob: 0.7 }, { id: 200, text: ' ran', prob: 0.2 }, { id: 201, text: ' was', prob: 0.1 },
    ] })
    events.push({ type: 'sample', cycle: c, chosen, method: 'top-k' })
    events.push({ type: 'append', cycle: c, token: chosen })
  }
  events.push({ type: 'run-end', reason: 'max-tokens' })
  return events
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
        distinctiveScore: ((k * 3 + 1) % 5) / 5,
      })
    }
  }
  return { type: 'attention-grid', layers, heads, cells }
}

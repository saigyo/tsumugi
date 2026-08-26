import type { TraceEvent, TokenInfo } from '../trace/types'

export function makeFixtureTrace(cycles = 2, layers = 3): TraceEvent[] {
  const events: TraceEvent[] = [
    { type: 'run-start', prompt: 'The cat', mode: 'sim', modelId: 'fixture',
      params: { temperature: 0.8, topK: 10, maxNewTokens: cycles } },
    { type: 'tokenize', tokens: [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }] },
  ]
  const words = [' sat', ' on', ' the', ' mat']
  for (let c = 0; c < cycles; c++) {
    const chosen: TokenInfo = { id: 100 + c, text: words[c % words.length] }
    events.push({ type: 'embed', cycle: c, seqLen: 2 + c, dims: 576,
      preview: [[0.1, -0.2, 0.3], [0.0, 0.5, -0.1]] })
    for (let l = 0; l < layers; l++) events.push({ type: 'layer', cycle: c, index: l, total: layers })
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

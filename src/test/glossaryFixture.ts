import type { RunSummary } from '../glossary/types'

export function sampleRun(over: Partial<RunSummary> = {}): RunSummary {
  return {
    mode: 'sim', modelId: 'fixture',
    params: { temperature: 0.8, topK: 10, maxNewTokens: 8 },
    tokens: [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }],
    cycle: 2, cycles: 8,
    stream: { seqLen: 4, dims: 576, layers: 30 },
    vocabSize: 49152,
    edits: { token: ' cat', layer: 29, ratio: 1.42 },
    votes: { candidate: ' sat', layer: 27, value: 2.31 },
    softmax: { temperature: 0.8, top: [
      { text: ' sat', logit: 9.1, prob: 0.7 }, { text: ' ran', logit: 7.2, prob: 0.2 }, { text: ' was', logit: 5.0, prob: 0.1 },
    ] },
    ...over,
  }
}

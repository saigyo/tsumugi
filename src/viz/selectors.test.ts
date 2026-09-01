import { expect, test } from 'vitest'
import { buildFixtureTrace, fixtureEmbedding, makeFixtureTrace } from '../test/fixtures'
import { activeStage, cycleTickIndices, distributionFor, embeddingRows, eventAt, flowShapes, latestOfType, stageEventIndex, visibleTokens } from './selectors'

const trace = makeFixtureTrace()  // 2 cycles, 3 layers

test('activeStage maps events to stage cards', () => {
  expect(activeStage({ type: 'tokenize', tokens: [] })).toBe('tokenizer')
  expect(activeStage(eventAt(trace, 2))).toBe('embeddings')
  expect(activeStage({ type: 'softmax', cycle: 0, temperature: 1, topK: [] })).toBe('logits')
  expect(activeStage({ type: 'append', cycle: 0, token: { id: 1, text: 'x' } })).toBe('sampler')
  expect(activeStage(undefined)).toBe(null)
  expect(activeStage({ type: 'run-end', reason: 'eos' })).toBe(null)
})

test('visibleTokens grows with cursor', () => {
  expect(visibleTokens(trace, 0).prompt).toHaveLength(0)          // before tokenize
  expect(visibleTokens(trace, 1).prompt).toHaveLength(2)          // after tokenize
  expect(visibleTokens(trace, 1).generated).toHaveLength(0)
  expect(visibleTokens(trace, trace.length - 1).generated).toHaveLength(2)
})

test('latestOfType finds most recent event at or before cursor', () => {
  const last = trace.length - 1
  expect(latestOfType(trace, last, 'softmax')?.cycle).toBe(1)
  expect(latestOfType(trace, 5, 'tokenize')?.type).toBe('tokenize')
  expect(latestOfType(trace, 0, 'softmax')).toBeUndefined()
})

test('cycleTickIndices marks append events', () => {
  const ticks = cycleTickIndices(trace)
  expect(ticks).toHaveLength(2)
  expect(trace[ticks[0]].type).toBe('append')
})

test('visibleTokens tags generated tokens with their cycle', () => {
  const { generated } = visibleTokens(trace, trace.length - 1)
  expect(generated.map((g) => g.cycle)).toEqual([0, 1])
})

test('distributionFor returns the softmax and sample of a cycle', () => {
  const d = distributionFor(trace, 1)
  expect(d?.softmax.cycle).toBe(1)
  expect(d?.softmax.topK[0].prob).toBe(0.7)
  expect(d?.sample.chosen.text).toBe(' on')
  expect(distributionFor(trace, 99)).toBeUndefined()
})

test('attention events map to the layers stage', () => {
  expect(activeStage({ type: 'attention', cycle: 0, heads: [] })).toBe('layers')
})

test('flowShapes are absent before the run reaches each stage', () => {
  expect(flowShapes(trace, -1)).toEqual({})
  const atTokenize = flowShapes(trace, 1)
  expect(atTokenize.ids).toBe('[2]')
  expect(atTokenize.stream).toBeUndefined()
})

test('flowShapes carry live tensor shapes once data exists', () => {
  const s = flowShapes(trace, 7)  // cycle 0 logits
  expect(s.ids).toBe('[2]')
  expect(s.stream).toBe('[2×576]')
  expect(s.lastRow).toBe('[1×576]')
  expect(s.vocab).toBe('[49 152]')
  expect(s.loop).toBe('+1 token')
})

test('flowShapes stream grows with the sequence', () => {
  expect(flowShapes(trace, 11).stream).toBe('[3×576]')  // cycle 1 embed
})

// fixture indices: 1=tokenize; c0: 2 embed, 3-5 layers, 6 attention, 7 logits, 8 softmax, 9 sample, 10 append;
// c1: 11 embed, 12-14 layers, 15 attention, 16 logits, 17 softmax, 18 sample, 19 append
test('stageEventIndex seeks to the representative event of the current cycle', () => {
  expect(stageEventIndex(trace, 4, 'tokenizer')).toBe(1)
  expect(stageEventIndex(trace, 4, 'layers')).toBe(6)      // cursor in cycle 0 → its attention
  expect(stageEventIndex(trace, 12, 'layers')).toBe(15)    // cursor in cycle 1
  expect(stageEventIndex(trace, 12, 'logits')).toBe(17)    // softmax of cycle 1
  expect(stageEventIndex(trace, 4, 'sampler')).toBe(9)
  expect(stageEventIndex(trace, 0, 'embeddings')).toBe(2)  // before first cycle → cycle 0
})

test('stageEventIndex falls back to the latest existing event when the cycle lacks one', () => {
  const partial = trace.slice(0, 13)  // cycle 1 cut off mid-layers
  expect(stageEventIndex(partial, 12, 'layers')).toBe(6)   // falls back to cycle 0 attention
  expect(stageEventIndex(partial, 12, 'sampler')).toBe(9)
})

test('stageEventIndex returns -1 when no event exists at all', () => {
  expect(stageEventIndex(trace.slice(0, 2), 1, 'layers')).toBe(-1)
})

// fixture indices: 2 = cycle-0 embed, 10 = cycle-0 append, 11 = cycle-1 embed
test('embeddingRows: asset-source runs return tokens only', () => {
  const r = embeddingRows(makeFixtureTrace(), 2)
  expect(r.source).toBe('asset')
  expect(r.rows).toBeUndefined()
  expect(r.tokens.map((t) => t.id)).toEqual([10, 11])
})

test('embeddingRows: model-source runs return one row per visible token', () => {
  const t = buildFixtureTrace({ embedRows: true })
  const c0 = embeddingRows(t, 2)
  expect(c0.source).toBe('model')
  expect(c0.rows).toHaveLength(2)
  expect(c0.rows?.[1]).toEqual(fixtureEmbedding(11))
  const c1 = embeddingRows(t, 11)
  expect(c1.tokens).toHaveLength(3)
  expect(c1.rows).toHaveLength(3)
  expect(c1.rows?.[2]).toEqual(fixtureEmbedding(100))
})

test('embeddingRows: a later asset-source cycle degrades the whole run to asset', () => {
  const t = buildFixtureTrace({ embedRows: true })
  const e = t[11]
  if (e.type === 'embed') { e.source = 'asset'; delete e.rows }
  expect(embeddingRows(t, 2).source).toBe('model')
  expect(embeddingRows(t, 11).source).toBe('asset')
})

test('embeddingRows: a token without a known row yet (cursor on append) falls back to asset', () => {
  const t = buildFixtureTrace({ embedRows: true })
  const r = embeddingRows(t, 10)
  expect(r.tokens).toHaveLength(3)
  expect(r.source).toBe('asset')
})

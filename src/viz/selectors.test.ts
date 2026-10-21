import { expect, test } from 'vitest'
import { buildFixtureTrace, fixtureEmbedding, makeFixtureTrace } from '../test/fixtures'
import { activeStage, chatLayout, cycleNeurons, cycleTickIndices, cycleVotes, distributionFor, editRows, embeddingRows, eventAt, flowShapes, latestOfType, stageEventIndex, viewedCycle, visibleTokens } from './selectors'
import type { TraceEvent } from '../trace/types'
import { validateTrace } from '../trace/validate'

const trace = makeFixtureTrace()  // 2 cycles, 3 layers

test('activeStage maps events to stage cards', () => {
  expect(activeStage({ type: 'tokenize', tokens: [] })).toBe('tokenizer')
  expect(activeStage(eventAt(trace, 2))).toBe('embeddings')
  expect(activeStage({ type: 'logits', cycle: 0, topK: [] })).toBe('logits')
  expect(activeStage({ type: 'softmax', cycle: 0, temperature: 1, topK: [] })).toBe('softmax')
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
  expect(s.probs).toBe('[10]')       // Softmax → Sampler: the top-k the run keeps (run-start params)
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
  expect(stageEventIndex(trace, 12, 'logits')).toBe(16)    // logits of cycle 1 — the scores, not the softmax
  expect(stageEventIndex(trace, 12, 'softmax')).toBe(17)   // softmax of cycle 1
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

// mlp fixture (2 prompt tokens, 3 layers, 2 cycles), indices:
// 0 run-start, 1 tokenize, 2 embed c0, 3..5 layers, 6 attention, 7 edits, 8 logits, 9 softmax, 10 sample, 11 append,
// 12 embed c1, 13..15 layers, 16 attention, 17 edits, 18 logits, …
const mlpTrace = buildFixtureTrace({ mlp: true })

test('editRows: on cycle 0 the prompt rows, looked up even before the edits event is reached', () => {
  const r = editRows(mlpTrace, 3)!
  expect(r.tokens.map((t) => t.text)).toEqual(['The', ' cat'])
  expect(r.attn).toHaveLength(2)
  expect(r.mlp[1]).toHaveLength(3)
})

test('editRows: on cycle 1 the generated token gets its row', () => {
  const r = editRows(mlpTrace, 13)!
  expect(r.tokens.map((t) => t.text)).toEqual(['The', ' cat', ' sat'])
  expect(r.attn).toHaveLength(3)
})

test('editRows: never shows rows past the viewed cycle', () => {
  expect(editRows(mlpTrace, 8)!.attn).toHaveLength(2)
})

test('editRows: undefined without edits events', () => {
  expect(editRows(buildFixtureTrace(), 4)).toBeUndefined()
})

test('cycleVotes: ok on the layers stage (looks ahead to the cycle’s logits), pending before it exists, none without votes', () => {
  const ok = cycleVotes(mlpTrace, 4)
  expect(ok.status).toBe('ok')
  if (ok.status === 'ok') expect(ok.candidates[0].votes.mlp).toHaveLength(3)
  expect(cycleVotes(mlpTrace.slice(0, 7), 4)).toEqual({ status: 'pending' })
  expect(cycleVotes(buildFixtureTrace(), 4)).toEqual({ status: 'none' })
})

test('cycleVotes: none when any candidate lacks votes', () => {
  const t = buildFixtureTrace({ mlp: true })
  const lg = t[8] as Extract<TraceEvent, { type: 'logits' }>
  const { votes: _dropped, ...rest } = lg.topK[2]
  void _dropped
  lg.topK[2] = rest
  expect(cycleVotes(t, 4)).toEqual({ status: 'none' })
})

test('viewedCycle follows the cursor', () => {
  expect(viewedCycle(mlpTrace, 4)).toBe(0)
  expect(viewedCycle(mlpTrace, 14)).toBe(1)
})

test('cycleNeurons: ok with data, none for older runs, pending while the cycle runs', () => {
  const real = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
  const layerAt = real.findIndex((e) => e.type === 'layer')
  expect(cycleNeurons(real, layerAt).status).toBe('ok')
  const old = buildFixtureTrace({ mode: 'real', mlp: true })
  expect(cycleNeurons(old, old.findIndex((e) => e.type === 'layer')).status).toBe('none')
  const running = real.slice(0, layerAt + 1)
  expect(cycleNeurons(running, layerAt).status).toBe('pending')
})

test('the trace fixture with neurons validates and its votes still add up to the logits', () => {
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
  expect(validateTrace(events)).toEqual([])
  const logits = events.find((e): e is Extract<TraceEvent, { type: 'logits' }> => e.type === 'logits')!
  for (const c of logits.topK) {
    const v = c.votes!
    expect(v.embed + v.attn.reduce((a, b) => a + b, 0) + v.mlp.reduce((a, b) => a + b, 0)).toBeCloseTo(c.logit, 5)
  }
})

test('chatLayout returns the run’s spans, or undefined for a plain-text run', () => {
  expect(chatLayout(buildFixtureTrace({ chat: true }))).toEqual({ system: [1, 21], user: [24, 31] })
  expect(chatLayout(buildFixtureTrace())).toBeUndefined()
})

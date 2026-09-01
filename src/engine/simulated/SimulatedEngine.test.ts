import { expect, test } from 'vitest'
import type { TraceEvent } from '../../trace/types'
import { validateTrace } from '../../trace/validate'
import { fakeTokenizer } from '../tokenizer'
import { SimulatedEngine } from './SimulatedEngine'

const PARAMS = { temperature: 0.8, topK: 10, maxNewTokens: 4 }

async function collect(prompt: string, params = PARAMS): Promise<TraceEvent[]> {
  const engine = new SimulatedEngine(fakeTokenizer(), { layers: 3 })
  const events: TraceEvent[] = []
  await engine.run(prompt, params, (e) => events.push(e)).done
  return events
}

test('produces a valid trace', async () => {
  expect(validateTrace(await collect('The cat sat'))).toEqual([])
})

test('run-start carries the simulated model vocab size', async () => {
  const events = await collect('Hi')
  const start = events[0]
  expect(start.type === 'run-start' && start.vocabSize).toBe(49152)
})

test('same prompt → identical trace (deterministic)', async () => {
  expect(await collect('Hello world')).toEqual(await collect('Hello world'))
})

test('different prompts → different traces', async () => {
  expect(JSON.stringify(await collect('aaa'))).not.toBe(JSON.stringify(await collect('bbb')))
})

test('temperature 0 uses greedy and picks the top candidate', async () => {
  const events = await collect('The cat', { ...PARAMS, temperature: 0 })
  const sample = events.find((e) => e.type === 'sample')
  const sm = events.find((e) => e.type === 'softmax')
  if (sample?.type !== 'sample' || sm?.type !== 'softmax') throw new Error('missing events')
  expect(sample.method).toBe('greedy')
  expect(sample.chosen.id).toBe(sm.topK[0].id)
})

test('respects maxNewTokens', async () => {
  const events = await collect('Hi', { ...PARAMS, maxNewTokens: 2 })
  expect(events.filter((e) => e.type === 'append').length).toBeLessThanOrEqual(2)
})

test('abort emits run-end aborted', async () => {
  const engine = new SimulatedEngine(fakeTokenizer(), { layers: 3 })
  const events: TraceEvent[] = []
  const handle = engine.run('Hi', { ...PARAMS, maxNewTokens: 50 }, (e) => events.push(e))
  handle.abort()
  await handle.done
  const last = events[events.length - 1]
  expect(last.type === 'run-end' && last.reason === 'aborted').toBe(true)
})

test('topK limits the number of candidates in logits/softmax', async () => {
  const events = await collect('The cat', { ...PARAMS, topK: 3 })
  const logits = events.find((e) => e.type === 'logits')
  const sm = events.find((e) => e.type === 'softmax')
  if (logits?.type !== 'logits' || sm?.type !== 'softmax') throw new Error('missing events')
  expect(logits.topK).toHaveLength(3)
  expect(sm.topK).toHaveLength(3)
})

test('topK is clamped to [1, 10]', async () => {
  const tooLow = (await collect('Hi', { ...PARAMS, topK: 0 })).find((e) => e.type === 'logits')
  const tooHigh = (await collect('Hi', { ...PARAMS, topK: 99 })).find((e) => e.type === 'logits')
  if (tooLow?.type !== 'logits' || tooHigh?.type !== 'logits') throw new Error('missing events')
  expect(tooLow.topK).toHaveLength(1)
  expect(tooHigh.topK).toHaveLength(10)
})

test('embed events are asset-sourced and carry no vectors (sim stays instant)', async () => {
  const events = await collect('one two three four five six')
  const embeds = events.filter((e) => e.type === 'embed')
  expect(embeds.length).toBeGreaterThan(0)
  for (const e of embeds) {
    if (e.type !== 'embed') continue
    expect(e.source).toBe('asset')
    expect(e.rows).toBeUndefined()
    expect(e.dims).toBe(576)
  }
})

test('emits one attention event per cycle, after layers and before logits', async () => {
  const events = await collect('The cat sat')
  const attn = events.filter((e) => e.type === 'attention')
  const appends = events.filter((e) => e.type === 'append')
  expect(attn.length).toBe(appends.length)
  const firstAttn = events.findIndex((e) => e.type === 'attention')
  expect(events[firstAttn - 1].type).toBe('layer')
  expect(events[firstAttn + 1].type).toBe('logits')
})

test('attention matrices grow with the sequence across cycles', async () => {
  const events = await collect('a b c')
  const attn = events.filter((e) => e.type === 'attention')
  if (attn[0]?.type !== 'attention' || attn[1]?.type !== 'attention') throw new Error('missing attention')
  expect(attn[1].heads[0].matrix.length).toBe(attn[0].heads[0].matrix.length + 1)
})

test('coreference example prompt yields a coreference head', async () => {
  const events = await collect('The cat sat on the mat because it was tired')
  const attn = events.find((e) => e.type === 'attention')
  if (attn?.type !== 'attention') throw new Error('missing attention')
  expect(attn.heads.some((h) => h.label === 'coreference')).toBe(true)
})

test('curated example runs follow their scripted continuation and end with eos', async () => {
  const events = await collect('The cat sat on the mat because it was tired', { ...PARAMS, maxNewTokens: 20 })
  const generated = events.filter((e) => e.type === 'append').map((e) => e.type === 'append' ? e.token.text : '')
  expect(generated.join('')).toBe(' It closed its eyes and fell asleep')
  const last = events[events.length - 1]
  expect(last.type === 'run-end' && last.reason === 'eos').toBe(true)
})

test('scripted tokens are the dominant top candidate each cycle', async () => {
  const events = await collect('one two three one two three one', { ...PARAMS, maxNewTokens: 20 })
  const logits = events.filter((e) => e.type === 'logits')
  const samples = events.filter((e) => e.type === 'sample')
  expect(logits.length).toBeGreaterThan(0)
  logits.forEach((l, i) => {
    if (l.type !== 'logits' || samples[i]?.type !== 'sample') throw new Error('shape')
    expect(l.topK[0].id).toBe(samples[i].chosen.id)          // scripted token is argmax
    expect(l.topK[0].logit - l.topK[1].logit).toBeGreaterThan(1.5)  // clearly dominant
  })
})

test('scripted runs still produce valid traces and respect maxNewTokens', async () => {
  const events = await collect('The cat sat on the mat because it was tired', { ...PARAMS, maxNewTokens: 3 })
  expect(validateTrace(events)).toEqual([])
  expect(events.filter((e) => e.type === 'append')).toHaveLength(3)
  const last = events[events.length - 1]
  expect(last.type === 'run-end' && last.reason === 'max-tokens').toBe(true)
})

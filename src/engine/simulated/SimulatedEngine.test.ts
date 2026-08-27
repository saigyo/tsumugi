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

test('embed preview is capped at 4 tokens × 16 dims', async () => {
  const events = await collect('one two three four five six')
  const embed = events.find((e) => e.type === 'embed')
  if (embed?.type !== 'embed') throw new Error('missing embed')
  expect(embed.preview.length).toBeLessThanOrEqual(4)
  expect(embed.preview[0]).toHaveLength(16)
})

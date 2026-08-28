import { expect, test, vi } from 'vitest'
import type { TraceEvent } from '../../trace/types'
import type { WorkerRequest, WorkerResponse } from './protocol'
import { TransformersEngine } from './TransformersEngine'

class FakeWorker {
  sent: WorkerRequest[] = []
  onmessage: ((e: MessageEvent<WorkerResponse>) => void) | null = null
  postMessage(msg: WorkerRequest) { this.sent.push(msg) }
  respond(msg: WorkerResponse) { this.onmessage?.({ data: msg } as MessageEvent<WorkerResponse>) }
  terminate() {}
}

function make() {
  const worker = new FakeWorker()
  const engine = new TransformersEngine(() => worker as unknown as Worker)
  return { worker, engine }
}

test('prepare resolves on ready and records device', async () => {
  const { worker, engine } = make()
  const onProgress = vi.fn()
  const p = engine.prepare(onProgress)
  worker.respond({ type: 'progress', info: { file: 'model.onnx', loaded: 1, total: 2 } })
  worker.respond({ type: 'ready', device: 'wasm', attentions: false })
  await p
  expect(onProgress).toHaveBeenCalledOnce()
  expect(engine.device).toBe('wasm')
  expect(worker.sent[0]).toMatchObject({ type: 'prepare' })
})

test('prepare rejects on fatal', async () => {
  const { worker, engine } = make()
  const p = engine.prepare()
  worker.respond({ type: 'fatal', message: 'download failed' })
  await expect(p).rejects.toThrow('download failed')
})

test('run forwards matching trace events and resolves on done', async () => {
  const { worker, engine } = make()
  const events: TraceEvent[] = []
  const handle = engine.run('Hi', { temperature: 1, topK: 10, maxNewTokens: 2 }, (e) => events.push(e))
  const runId = (worker.sent.at(-1) as Extract<WorkerRequest, { type: 'run' }>).runId
  worker.respond({ type: 'trace', runId, event: { type: 'run-end', reason: 'max-tokens' } })
  worker.respond({ type: 'trace', runId: runId + 99, event: { type: 'run-end', reason: 'eos' } })  // ignored
  worker.respond({ type: 'done', runId })
  await handle.done
  expect(events).toHaveLength(1)
})

test('fatal mid-run emits synthetic error run-end', async () => {
  const { worker, engine } = make()
  const events: TraceEvent[] = []
  const handle = engine.run('Hi', { temperature: 1, topK: 10, maxNewTokens: 2 }, (e) => events.push(e))
  worker.respond({ type: 'fatal', message: 'worker crashed' })
  await handle.done
  expect(events.at(-1)).toMatchObject({ type: 'run-end', reason: 'error', message: 'worker crashed' })
})

test('abort posts abort message', () => {
  const { worker, engine } = make()
  const handle = engine.run('Hi', { temperature: 1, topK: 10, maxNewTokens: 2 }, () => {})
  handle.abort()
  expect(worker.sent.some((m) => m.type === 'abort')).toBe(true)
})

test('ready records the attentions capability', async () => {
  const { worker, engine } = make()
  const p = engine.prepare()
  worker.respond({ type: 'ready', device: 'webgpu', attentions: true })
  await p
  expect(engine.attentions).toBe(true)
})

test('fetchHead posts head-request and resolves on the matching response', async () => {
  const { worker, engine } = make()
  const p = engine.fetchHead(2, 5)
  expect(worker.sent.at(-1)).toEqual({ type: 'head-request', layer: 2, head: 5 })
  worker.respond({ type: 'head-response', layer: 1, head: 5, matrix: [[1]], label: null, score: null })  // ignored
  worker.respond({ type: 'head-response', layer: 2, head: 5, matrix: [[1], [0.5, 0.5]], label: 'previous-token', score: 0.5 })
  const r = await p
  expect(r.matrix).toEqual([[1], [0.5, 0.5]])
  expect(r.label).toBe('previous-token')
  expect(r.score).toBe(0.5)
})

test('fetchHead resolves the unavailable case as an empty matrix', async () => {
  const { worker, engine } = make()
  const p = engine.fetchHead(0, 0)
  worker.respond({ type: 'head-response', layer: 0, head: 0, matrix: [], label: null, score: null })
  const r = await p
  expect(r.matrix).toEqual([])
  expect(r.label).toBeNull()
})

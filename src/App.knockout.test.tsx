import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'
import { useRunsStore, type RunRecord } from './app/runsStore'
import { initialPlayerState } from './player/reducer'
import { usePlayerStore } from './player/store'
import { buildFixtureTrace, makeRunRecord } from './test/fixtures'
import { useTraceStore } from './trace/store'
import type { KnockoutLayer, KnockoutSweep, TraceEvent } from './trace/types'
import { useKnockoutRequest } from './viz/knockoutRequest'

interface FakeSweep {
  req: { ids: number[]; chosenId: number; cycle: number; seed: number }
  onBaseline: (p: number) => void
  onLayer: (e: KnockoutLayer) => void
  finish: (ok: boolean, error?: string) => void
  aborted: boolean
}
let sweeps: FakeSweep[] = []
let runs: Array<{ args: unknown[]; emit: (e: TraceEvent) => void }> = []
let sweepMode: 'ok' | 'throw' | 'reject' = 'ok'
let knockoutCapable = true
let prepare: { resolve: () => void } | null = null

vi.mock('./engine/tokenizer', async (importOriginal) => {
  const mod = await importOriginal<typeof import('./engine/tokenizer')>()
  return { ...mod, loadTokenizer: async () => mod.fallbackTokenizer() }
})

vi.mock('./engine/transformers/TransformersEngine', () => {
  class FakeTransformersEngine {
    device = 'wasm' as const
    get knockout() { return knockoutCapable }
    revision = 'rev-1'
    prepare() { return new Promise<void>((resolve) => { prepare = { resolve } }) }
    run(...args: unknown[]) {
      runs.push({ args, emit: args[2] as (e: TraceEvent) => void })
      return { abort() {}, done: Promise.resolve() }
    }
    sweep(req: FakeSweep['req'], onBaseline: FakeSweep['onBaseline'], onLayer: FakeSweep['onLayer']) {
      if (sweepMode === 'throw') throw new Error('sweep failed')
      let settle!: (r: { ok: boolean; error?: string }) => void
      let fail!: (e: Error) => void
      const done = new Promise<{ ok: boolean; error?: string }>((r, j) => { settle = r; fail = j })
      const finish = (ok: boolean, error?: string) => settle({ ok, ...(error !== undefined ? { error } : {}) })
      if (sweepMode === 'reject') queueMicrotask(() => fail(new Error('worker died')))
      const s: FakeSweep = { req, onBaseline, onLayer, finish, aborted: false }
      sweeps.push(s)
      return { abort: () => { s.aborted = true; finish(false) }, done }
    }
  }
  return { TransformersEngine: FakeTransformersEngine }
})

const layer = (l: number): KnockoutLayer => ({ layer: l, prob: 0.1, top: { id: 7, text: 'x', prob: 0.5 }, flips: true })

function seedRecords(): { a: RunRecord; b: RunRecord } {
  const a = makeRunRecord(1, { mode: 'real', cycles: 3 })
  const b = makeRunRecord(2, { mode: 'real', cycles: 3 })
  useRunsStore.setState({ records: [a, b], activeId: a.id, nextSeq: 3 })
  return { a, b }
}

async function ready() {
  render(<App />)
  fireEvent.click(screen.getByTestId('mode-real'))
  await act(async () => { prepare!.resolve() })
  await waitFor(() => expect(useKnockoutRequest.getState().measure).not.toBeNull())
}

beforeEach(() => {
  sweeps = []; runs = []; prepare = null; sweepMode = 'ok'; knockoutCapable = true
  localStorage.clear()
  useTraceStore.getState().clear()
  usePlayerStore.setState({ ...initialPlayerState })
  useRunsStore.setState({ records: [], activeId: null, nextSeq: 1 })
})
afterEach(() => cleanup())

test('the model capabilities and stable handlers are published once the real model is ready', async () => {
  render(<App />)
  expect(useKnockoutRequest.getState().model).toBeNull()
  fireEvent.click(screen.getByTestId('mode-real'))
  await act(async () => { prepare!.resolve() })
  await waitFor(() => expect(useKnockoutRequest.getState().model).toEqual({ knockout: true, revision: 'rev-1' }))
  const { measure, runKnockout } = useKnockoutRequest.getState()
  fireEvent.click(screen.getByTestId('mode-sim'))   // re-renders App
  expect(useKnockoutRequest.getState().measure).toBe(measure)
  expect(useKnockoutRequest.getState().runKnockout).toBe(runKnockout)
})

test('measuring sends the prompt and earlier appended tokens, disables Generate and shows progress', async () => {
  const { a } = seedRecords()
  await ready()
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'hi' } })
  act(() => useKnockoutRequest.getState().measure!(1))
  expect(sweeps).toHaveLength(1)
  const tokenize = a.events.find((e) => e.type === 'tokenize')!
  const promptIds = tokenize.type === 'tokenize' ? tokenize.tokens.map((t) => t.id) : []
  expect(sweeps[0].req.ids).toEqual([...promptIds, 100])
  expect(sweeps[0].req.cycle).toBe(1)
  expect(sweeps[0].req.chosenId).toBe(101)
  expect(screen.getByTestId('btn-generate')).toBeDisabled()
  expect(screen.getByTestId('btn-generate')).toHaveTextContent('Measuring…')
  act(() => { sweeps[0].onBaseline(0.7); sweeps[0].onLayer(layer(0)) })
  expect(useKnockoutRequest.getState().running).toMatchObject({ recordId: a.id, cycle: 1, baseline: 0.7 })
  expect(useKnockoutRequest.getState().running!.layers).toHaveLength(1)
})

test('a completed sweep lands on the active record and clears the running state', async () => {
  const { a, b } = seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().measure!(0))
  act(() => { sweeps[0].onBaseline(0.7); sweeps[0].onLayer(layer(0)); sweeps[0].onLayer(layer(1)) })
  await act(async () => { sweeps[0].finish(true) })
  const saved = useRunsStore.getState().records.find((r) => r.id === a.id)!.sweeps as KnockoutSweep[]
  expect(saved).toHaveLength(1)
  expect(saved[0]).toMatchObject({ cycle: 0, revision: 'rev-1', baseline: 0.7, chosen: { id: 100 } })
  expect(saved[0].layers.map((l) => l.layer)).toEqual([0, 1])
  expect(useRunsStore.getState().records.find((r) => r.id === b.id)!.sweeps).toBeUndefined()
  expect(useKnockoutRequest.getState().running).toBeNull()
  expect(screen.getByTestId('btn-generate')).toHaveTextContent('Generate')
})

test('Stop during a sweep saves nothing', async () => {
  const { a } = seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().measure!(0))
  await act(async () => { fireEvent.click(screen.getByTestId('btn-stop')) })
  expect(sweeps[0].aborted).toBe(true)
  expect(useRunsStore.getState().records.find((r) => r.id === a.id)!.sweeps).toBeUndefined()
  expect(useKnockoutRequest.getState().running).toBeNull()
  expect(screen.getByTestId('btn-generate')).not.toHaveTextContent('Measuring…')
})

test('activating another run mid-sweep aborts it and saves nothing', async () => {
  const { a, b } = seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().measure!(0))
  fireEvent.click(screen.getAllByTestId('run-chip-main')[1])
  await waitFor(() => expect(useRunsStore.getState().activeId).toBe(b.id))
  expect(sweeps[0].aborted).toBe(true)
  await act(async () => { sweeps[0].finish(true) })
  for (const id of [a.id, b.id]) expect(useRunsStore.getState().records.find((r) => r.id === id)!.sweeps).toBeUndefined()
})

test('a knockout at cycle 0 sends forced [] and opens Compare at that cycle when the run seals', async () => {
  const { a } = seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().runKnockout!(3, 0))
  await waitFor(() => expect(runs).toHaveLength(1))
  const [prompt, params, , format, options] = runs[0].args as [string, unknown, unknown, unknown, { knockout: unknown }]
  expect(prompt).toBe('The cat')
  expect(params).toEqual(a.meta.params)
  expect(format).toEqual({ kind: 'text' })
  expect(options.knockout).toMatchObject({ forced: [], origin: { layer: 3, cycle: 0, sourceId: a.id } })
  const trace = buildFixtureTrace({ mode: 'real', cycles: 3 })
  act(() => {
    trace.forEach((e, i) => runs[0].emit(i === 0 && e.type === 'run-start'
      ? { ...e, knockout: { layer: 3, cycle: 0, sourceId: a.id } } : e))
  })
  await waitFor(() => expect(screen.getAllByTestId('cmp-tick').length).toBeGreaterThan(0))
  const newest = useRunsStore.getState().records.at(-1)!
  expect(newest.id).not.toBe(a.id)
  expect(newest.meta.knockout).toMatchObject({ layer: 3, sourceId: a.id })
  expect(screen.getAllByTestId('cmp-tick')[0].dataset.selected).toBe('true')
})

test('a knockout at a later cycle forces the earlier appended tokens', async () => {
  seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().runKnockout!(2, 2))
  await waitFor(() => expect(runs).toHaveLength(1))
  const options = runs[0].args[4] as { knockout: { forced: number[]; seed: number } }
  expect(options.knockout.forced).toEqual([100, 101])
})

test('a sweep that throws leaves Generate usable and saves nothing', async () => {
  const { a } = seedRecords()
  await ready()
  sweepMode = 'throw'
  act(() => useKnockoutRequest.getState().measure!(0))
  expect(useKnockoutRequest.getState().running).toBeNull()
  expect(screen.getByTestId('btn-generate')).not.toHaveTextContent('Measuring…')
  expect(useRunsStore.getState().records.find((r) => r.id === a.id)!.sweeps).toBeUndefined()
})

test('a sweep whose done rejects clears the state, saves nothing and does not block Generate', async () => {
  const { a } = seedRecords()
  await ready()
  sweepMode = 'reject'
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'hi' } })
  act(() => useKnockoutRequest.getState().measure!(0))
  await waitFor(() => expect(useKnockoutRequest.getState().running).toBeNull())
  expect(screen.getByTestId('btn-generate')).not.toBeDisabled()
  expect(useRunsStore.getState().records.find((r) => r.id === a.id)!.sweeps).toBeUndefined()
  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => expect(runs).toHaveLength(1))
})

test('during a sweep Generate is disabled, a knockout request is ignored, and Stop saves nothing', async () => {
  const { a } = seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().measure!(0))
  expect(screen.getByTestId('btn-generate')).toBeDisabled()
  act(() => useKnockoutRequest.getState().runKnockout!(1, 0))   // ignored while measuring
  expect(runs).toHaveLength(0)
  await act(async () => { fireEvent.click(screen.getByTestId('btn-stop')) })
  expect(useRunsStore.getState().records.find((r) => r.id === a.id)!.sweeps).toBeUndefined()
})

test('stopping a knockout run keeps the sealed record but does not open Compare', async () => {
  const { a } = seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().runKnockout!(3, 0))
  await waitFor(() => expect(runs).toHaveLength(1))
  const trace = buildFixtureTrace({ mode: 'real', cycles: 1, reason: 'aborted' })
  act(() => {
    trace.forEach((e, i) => runs[0].emit(i === 0 && e.type === 'run-start'
      ? { ...e, knockout: { layer: 3, cycle: 0, sourceId: a.id } } : e))
  })
  await waitFor(() => expect(useRunsStore.getState().records).toHaveLength(3))
  expect(useRunsStore.getState().records.at(-1)!.meta.reason).toBe('aborted')
  expect(screen.queryAllByTestId('cmp-tick')).toHaveLength(0)
})

test('a failed sweep stores the message, saves nothing and leaves Measure usable', async () => {
  const { a } = seedRecords()
  await ready()
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'hi' } })
  act(() => useKnockoutRequest.getState().measure!(1))
  act(() => { sweeps[0].onBaseline(0.7); sweeps[0].onLayer(layer(0)) })
  await act(async () => { sweeps[0].finish(false, 'out of memory') })
  expect(useKnockoutRequest.getState().error).toEqual({ recordId: a.id, cycle: 1, message: 'out of memory' })
  expect(useKnockoutRequest.getState().running).toBeNull()
  expect(useRunsStore.getState().records.find((r) => r.id === a.id)!.sweeps).toBeUndefined()
  expect(screen.getByTestId('btn-generate')).not.toBeDisabled()
  // a new measurement clears the message
  act(() => useKnockoutRequest.getState().measure!(1))
  expect(sweeps).toHaveLength(2)
  expect(useKnockoutRequest.getState().error).toBeNull()
})

test('an aborted sweep sets no error', async () => {
  seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().measure!(0))
  await act(async () => { fireEvent.click(screen.getByTestId('btn-stop')) })
  expect(useKnockoutRequest.getState().error).toBeNull()
})

test('activating another run clears a sweep error', async () => {
  const { b } = seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().measure!(0))
  await act(async () => { sweeps[0].finish(false, 'boom') })
  expect(useKnockoutRequest.getState().error).not.toBeNull()
  fireEvent.click(screen.getAllByTestId('run-chip-main')[1])
  await waitFor(() => expect(useRunsStore.getState().activeId).toBe(b.id))
  expect(useKnockoutRequest.getState().error).toBeNull()
})

test('a synchronous sweep throw goes to the card error, not the model banner', async () => {
  const { a } = seedRecords()
  await ready()
  sweepMode = 'throw'
  act(() => useKnockoutRequest.getState().measure!(0))
  expect(useKnockoutRequest.getState().error).toEqual({ recordId: a.id, cycle: 0, message: 'sweep failed' })
  expect(screen.queryByTestId('model-error')).toBeNull()
})

test('a knockout run that ends in an error does not open Compare', async () => {
  const { a } = seedRecords()
  await ready()
  act(() => useKnockoutRequest.getState().runKnockout!(3, 0))
  await waitFor(() => expect(runs).toHaveLength(1))
  const trace = buildFixtureTrace({ mode: 'real', cycles: 1, reason: 'error' })
  act(() => {
    trace.forEach((e, i) => runs[0].emit(i === 0 && e.type === 'run-start'
      ? { ...e, knockout: { layer: 3, cycle: 0, sourceId: a.id } } : e))
  })
  await waitFor(() => expect(useRunsStore.getState().records).toHaveLength(3))
  expect(useRunsStore.getState().records.at(-1)!.meta.reason).toBe('error')
  expect(screen.queryAllByTestId('cmp-tick')).toHaveLength(0)
})

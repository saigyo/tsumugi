import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'
import { useTraceStore } from './trace/store'
import { initialPlayerState } from './player/reducer'
import { usePlayerStore } from './player/store'

vi.mock('./engine/tokenizer', async (importOriginal) => {
  const mod = await importOriginal<typeof import('./engine/tokenizer')>()
  return { ...mod, loadTokenizer: async () => mod.fallbackTokenizer() }
})

vi.mock('./engine/transformers/TransformersEngine', () => {
  // Constructing the real engine (or letting prepare() resolve) is exercised elsewhere;
  // here we only need a model that never finishes loading, to test the "not ready yet" gate.
  // (A plain class, not `vi.fn().mockImplementation(() => ({...}))`: vitest invokes the
  // implementation via `new`, and an arrow-function implementation can't be used as a
  // constructor.)
  class FakeTransformersEngine {
    device = null
    prepare() { return new Promise(() => {}) }
    run() { throw new Error('not used in this test') }
  }
  return { TransformersEngine: FakeTransformersEngine }
})

beforeEach(() => {
  useTraceStore.getState().clear()
  usePlayerStore.setState({ ...initialPlayerState })
})

afterEach(() => cleanup())

test('generate in sim mode fills the trace and plays', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => {
    const events = useTraceStore.getState().events
    expect(events.length).toBeGreaterThan(10)
    expect(events.at(-1)?.type).toBe('run-end')
  })
  const last = useTraceStore.getState().events.at(-1)
  expect(last?.type).toBe('run-end')
  expect(usePlayerStore.getState().status).toBe('playing')
})

test('clicking Generate during an active run aborts it and starts a fresh one', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })
  fireEvent.change(screen.getByTestId('maxtok-input'), { target: { value: '100' } })

  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => {
    expect(useTraceStore.getState().events.length).toBeGreaterThan(0)
  })

  // Re-click mid-run: the old run must be aborted and a new one started, not blocked.
  expect(screen.getByTestId('btn-generate')).not.toBeDisabled()
  fireEvent.click(screen.getByTestId('btn-generate'))

  await waitFor(() => {
    expect(useTraceStore.getState().events.at(-1)?.type).toBe('run-end')
  })

  const finalEvents = useTraceStore.getState().events
  expect(finalEvents.filter((e) => e.type === 'run-start')).toHaveLength(1)
})

test('generate is disabled while real mode is still loading the model', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })
  expect(screen.getByTestId('btn-generate')).not.toBeDisabled()

  fireEvent.click(screen.getByTestId('mode-toggle'))

  await waitFor(() => {
    expect(screen.getByTestId('btn-generate')).toBeDisabled()
  })
})

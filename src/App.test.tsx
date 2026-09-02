import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'
import { useTraceStore } from './trace/store'
import { initialPlayerState } from './player/reducer'
import { usePlayerStore } from './player/store'
import { useRunsStore } from './app/runsStore'
import { _resetArchiveForTests } from './app/runArchive'

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
  useRunsStore.setState({ records: [], activeId: null, nextSeq: 1, persistFailed: false })
  _resetArchiveForTests()
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

test('while a run is in flight, Generate is disabled and shows activity; it re-enables afterwards', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })
  fireEvent.change(screen.getByTestId('maxtok-input'), { target: { value: '100' } })

  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => {
    expect(useTraceStore.getState().events.length).toBeGreaterThan(0)
  })
  const btn = screen.getByTestId('btn-generate')
  expect(btn).toBeDisabled()
  expect(btn).toHaveTextContent('Generating…')
  expect(btn.querySelector('[data-testid="generate-activity"]')).not.toBeNull()
  expect(screen.getAllByTestId('example-chip')[0]).toBeDisabled()

  await waitFor(() => {
    expect(useTraceStore.getState().events.at(-1)?.type).toBe('run-end')
  })
  await waitFor(() => expect(screen.getByTestId('btn-generate')).not.toBeDisabled())
  expect(screen.getByTestId('btn-generate')).toHaveTextContent('Generate')
  expect(screen.queryByTestId('btn-stop')).toBeNull()
})

test('Stop aborts the run in flight; the trace ends with reason aborted and Generate is back', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })
  fireEvent.change(screen.getByTestId('maxtok-input'), { target: { value: '100' } })

  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => {
    expect(useTraceStore.getState().events.length).toBeGreaterThan(0)
  })
  fireEvent.click(screen.getByTestId('btn-stop'))

  await waitFor(() => {
    const last = useTraceStore.getState().events.at(-1)
    expect(last?.type === 'run-end' && last.reason).toBe('aborted')
  })
  await waitFor(() => expect(screen.getByTestId('btn-generate')).not.toBeDisabled())
  expect(useTraceStore.getState().events.filter((e) => e.type === 'run-start')).toHaveLength(1)
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

async function generateAndFinish(prompt: string) {
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: prompt } })
  // Generate is disabled while a run is in flight; a previous call may still be settling
  await waitFor(() => expect(screen.getByTestId('btn-generate')).not.toBeDisabled())
  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => {
    expect(useTraceStore.getState().events.at(-1)?.type).toBe('run-end')
  })
  await waitFor(() => expect(screen.getByTestId('btn-generate')).not.toBeDisabled())
}

test('completed runs are sealed onto the shelf', async () => {
  render(<App />)
  await generateAndFinish('one two three')
  await waitFor(() => expect(screen.getAllByTestId('run-chip')).toHaveLength(1))
  expect(screen.getByTestId('run-chip')).toHaveTextContent('#1 · one two three')
  expect(useRunsStore.getState().activeId).not.toBeNull()
})

test('activating an archived run reloads its trace, parked at run-end', async () => {
  render(<App />)
  await generateAndFinish('one two three')
  await generateAndFinish('red green blue')
  await waitFor(() => expect(screen.getAllByTestId('run-chip')).toHaveLength(2))
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])
  await waitFor(() => {
    const first = useTraceStore.getState().events[0]
    expect(first?.type === 'run-start' && first.prompt).toBe('one two three')
  })
  const { cursor, status } = usePlayerStore.getState()
  expect(cursor).toBe(useTraceStore.getState().events.length - 1)
  expect(status).toBe('paused')
})

test('compare arms from the shelf, opens the view, and exits', async () => {
  render(<App />)
  await generateAndFinish('one two three')
  await generateAndFinish('red green blue')
  await waitFor(() => expect(screen.getAllByTestId('run-chip')).toHaveLength(2))
  fireEvent.click(screen.getByTestId('btn-compare-arm'))
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])
  expect(screen.getByTestId('compare-view')).toBeInTheDocument()
  expect(screen.getByTestId('cmp-badge')).toHaveTextContent('different prompts')
  expect(screen.queryByTestId('stage-card')).toBeNull()          // player stack hidden
  fireEvent.click(screen.getByTestId('btn-compare-exit'))
  expect(screen.queryByTestId('compare-view')).toBeNull()
  expect(screen.getAllByTestId('stage-card').length).toBeGreaterThan(0)
})

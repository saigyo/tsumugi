import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'
import { useTraceStore } from './trace/store'
import { initialPlayerState } from './player/reducer'
import { usePlayerStore } from './player/store'
import { useRunsStore } from './app/runsStore'
import { _resetArchiveForTests, archiveImport } from './app/runArchive'
import { serializeRun } from './app/runFiles'
import { buildFixtureTrace } from './test/fixtures'
import { resetGlossaryStore } from './glossary/store'
import { useRunRequest } from './viz/runRequest'

// what the mocked engine does: never finish loading (null), or load as this revision
const engineMock = vi.hoisted(() => ({
  loadAs: null as { revision: string | undefined } | null,
  runs: [] as { prompt: string; params: { maxNewTokens: number }; options?: { seed?: number } }[],
}))

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
    revision: string | undefined = undefined
    prepare() {
      const load = engineMock.loadAs
      if (!load) return new Promise(() => {})
      this.revision = load.revision
      return Promise.resolve()
    }
    run(prompt: string, params: { maxNewTokens: number }, _onEvent: unknown, _format: unknown, options?: { seed?: number }) {
      engineMock.runs.push({ prompt, params, options })
      return { abort() {}, done: Promise.resolve() }
    }
  }
  return { TransformersEngine: FakeTransformersEngine }
})

beforeEach(() => {
  useTraceStore.getState().clear()
  usePlayerStore.setState({ ...initialPlayerState })
  useRunsStore.setState({ records: [], activeId: null, nextSeq: 1, persistFailed: false })
  _resetArchiveForTests()
  engineMock.loadAs = null
  engineMock.runs = []
})

afterEach(() => { cleanup(); resetGlossaryStore(); location.hash = '' })

test('the header carries the tagline', () => {
  render(<App />)
  expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Watch a language model write, one token at a time')
})

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
  expect(screen.queryByTestId('examples-row')).toBeNull()   // closed by the first run
  fireEvent.click(screen.getByTestId('toggle-examples'))
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

  fireEvent.click(screen.getByTestId('mode-real'))

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
  fireEvent.click(screen.getAllByTestId('run-chip-main')[1])
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
  fireEvent.click(screen.getAllByTestId('run-chip-main')[1])
  expect(screen.getByTestId('compare-view')).toBeInTheDocument()
  expect(screen.getByTestId('cmp-badge')).toHaveTextContent('different prompts')
  expect(screen.queryByTestId('stage-card')).toBeNull()          // player stack hidden
  fireEvent.click(screen.getByTestId('btn-compare-exit'))
  expect(screen.queryByTestId('compare-view')).toBeNull()
  expect(screen.getAllByTestId('stage-card').length).toBeGreaterThan(0)
})

test('the ? button opens the glossary drawer, which closes again', () => {
  render(<App />)
  fireEvent.click(screen.getByTestId('btn-help'))
  expect(screen.getByTestId('glossary-drawer')).toBeInTheDocument()
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Glossary')
  fireEvent.click(screen.getByTestId('gl-close'))
  expect(screen.queryByTestId('glossary-drawer')).toBeNull()
})

test('an archived real run shows its MLP cards while the app is in sim mode', async () => {
  const record = archiveImport({ meta: { prompt: 'The cat', params: { temperature: 0.8, topK: 10, maxNewTokens: 2 },
    mode: 'real', modelId: 'fixture', endedAt: 1, reason: 'max-tokens' },
    events: buildFixtureTrace({ mode: 'real', mlp: true }) })
  render(<App />)
  fireEvent.click(within(screen.getAllByTestId('run-chip').find((c) => c.textContent?.includes('The cat'))!).getByTestId('run-chip-main'))
  fireEvent.click((await screen.findAllByTestId('stage-card'))[2])
  expect((await screen.findAllByTestId('vote-cell', {}, { timeout: 2000 })).length).toBeGreaterThan(0)
  expect(record.id).toBeTruthy()
})

test('when the archive refuses an imported run, the shelf says why instead of blaming the file', async () => {
  const record = archiveImport({ meta: { prompt: 'The cat', params: { temperature: 0.8, topK: 10, maxNewTokens: 2 },
    mode: 'sim', endedAt: 1, reason: 'max-tokens' }, events: buildFixtureTrace() })
  const original = useRunsStore.getState().importRecord
  useRunsStore.setState({ importRecord: () => { throw new Error('storage quota exceeded') } })
  try {
    render(<App />)
    const file = new File([serializeRun(record).json], 'run.json', { type: 'application/json' })
    fireEvent.change(screen.getByTestId('import-input'), { target: { files: [file] } })
    expect(await screen.findByTestId('import-error')).toHaveTextContent('could not add the run: storage quota exceeded')
  } finally { useRunsStore.setState({ importRecord: original }) }
})

test('a share link fills the prompt bar and clears the hash', async () => {
  location.hash = '#run=1&p=Hello%20there&f=text&t=0.5&k=7&n=33&seed=42'
  render(<App />)
  expect(screen.getByTestId('prompt-input')).toHaveValue('Hello there')
  expect(screen.getByTestId('maxtok-input')).toHaveValue(33)
  await waitFor(() => expect(location.hash).toBe(''))
  // the link switches to the real model, which the mock never finishes loading
  await waitFor(() => expect(screen.getByTestId('btn-generate')).toBeDisabled())
  expect(useTraceStore.getState().events).toHaveLength(0)
})

const NOTE_TEXT = (made: string, loaded: string) =>
  `This link was made with model version ${made}; the loaded model is ${loaded}, so the text may differ.`

async function openLink(hash: string, loadedRevision: string) {
  engineMock.loadAs = { revision: loadedRevision }
  location.hash = hash
  render(<App />)
  await waitFor(() => expect(screen.getByTestId('btn-generate')).not.toBeDisabled())
}

test('a link made with another model version shows a note', async () => {
  await openLink('#run=1&p=Hi&f=text&t=0.5&k=7&n=33&seed=42&rev=v6', 'v7')
  expect(await screen.findByTestId('link-note')).toHaveTextContent(NOTE_TEXT('v6', 'v7'))
  expect(screen.getByTestId('link-note').textContent).toBe(NOTE_TEXT('v6', 'v7'))
})

test('a link without a version counts as stock', async () => {
  await openLink('#run=1&p=Hi&f=text&t=0.5&k=7&n=33&seed=42', 'v7')
  expect((await screen.findByTestId('link-note')).textContent).toBe(NOTE_TEXT('stock', 'v7'))
})

test('a link made with the loaded model version shows no note', async () => {
  await openLink('#run=1&p=Hi&f=text&t=0.5&k=7&n=33&seed=42&rev=v7', 'v7')
  expect(screen.queryByTestId('link-note')).toBeNull()
})

test('the note goes away on the next Generate', async () => {
  await openLink('#run=1&p=Hi&f=text&t=0.5&k=7&n=33&seed=42&rev=v6', 'v7')
  await screen.findByTestId('link-note')
  fireEvent.click(screen.getByTestId('btn-generate'))
  await waitFor(() => expect(screen.queryByTestId('link-note')).toBeNull())
})

test('Continue reruns the shown trace with its own seed after the active chip was removed', async () => {
  await openLink('#run=1&p=Hi&f=text&t=0.5&k=7&n=33&seed=42&rev=v7', 'v7')
  act(() => {
    useTraceStore.getState().append({ type: 'run-start', prompt: 'Once upon', mode: 'real', modelId: 'm', seed: 777,
      revision: 'v7', params: { temperature: 0.8, topK: 10, maxNewTokens: 20 } })
    useTraceStore.getState().append({ type: 'run-end', reason: 'max-tokens' })
  })
  expect(useRunsStore.getState().activeId).toBeNull()      // no shelf record is active
  await waitFor(() => expect(useRunRequest.getState().continueRun).toBeTruthy())
  await act(async () => { useRunRequest.getState().continueRun?.(40) })
  await waitFor(() => expect(engineMock.runs).toHaveLength(1))
  expect(engineMock.runs[0].prompt).toBe('Once upon')
  expect(engineMock.runs[0].params.maxNewTokens).toBe(40)
  expect(engineMock.runs[0].options).toEqual({ seed: 777 })
})

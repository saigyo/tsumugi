import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'
import { useTraceStore } from './trace/store'
import { initialPlayerState } from './player/reducer'
import { usePlayerStore } from './player/store'
import { useRealModel } from './viz/realModelRequest'

let constructCount = 0
let runs: unknown[][] = []
let cached = false
// set to hold the start-up cache check open until the test settles it
let cacheCheck: { promise: Promise<boolean>; resolve: (ok: boolean) => void } | null = null
let deferreds: Array<{ resolve: () => void; reject: (err: Error) => void }> = []

vi.mock('./engine/tokenizer', async (importOriginal) => {
  const mod = await importOriginal<typeof import('./engine/tokenizer')>()
  return { ...mod, loadTokenizer: async () => mod.fallbackTokenizer() }
})

vi.mock('./app/realModel', async (importOriginal) => {
  const mod = await importOriginal<typeof import('./app/realModel')>()
  return { ...mod, realModelCached: () => cacheCheck?.promise ?? Promise.resolve(cached) }
})

vi.mock('./engine/transformers/TransformersEngine', () => {
  // A prepare() that stays pending until the test explicitly settles it, so we can
  // toggle modes rapidly while a "download" is in flight and assert only one engine
  // (and thus one download / worker) was ever constructed.
  class FakeTransformersEngine {
    device = 'wasm' as const
    constructor() { constructCount++ }
    prepare() {
      return new Promise<void>((resolve, reject) => { deferreds.push({ resolve, reject }) })
    }
    run(...args: unknown[]) { runs.push(args); return { abort() {}, done: Promise.resolve() } }
  }
  return { TransformersEngine: FakeTransformersEngine }
})

beforeEach(() => {
  constructCount = 0
  deferreds = []
  runs = []
  cached = false
  cacheCheck = null
  localStorage.clear()
  useTraceStore.getState().clear()
  usePlayerStore.setState({ ...initialPlayerState })
})

afterEach(() => cleanup())

test('rapid mode toggling mid-prepare constructs exactly one TransformersEngine', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })

  // real -> sim -> real -> sim -> real, all while the first prepare() is still pending.
  fireEvent.click(screen.getByTestId('mode-real'))
  fireEvent.click(screen.getByTestId('mode-sim'))
  fireEvent.click(screen.getByTestId('mode-real'))
  fireEvent.click(screen.getByTestId('mode-sim'))
  fireEvent.click(screen.getByTestId('mode-real'))

  expect(constructCount).toBe(1)
  expect(deferreds).toHaveLength(1)

  deferreds[0].resolve()
  await waitFor(() => {
    expect(screen.getByTestId('btn-generate')).not.toBeDisabled()
  })

  expect(constructCount).toBe(1)
})

test('a failed prepare clears the in-flight guard so a later toggle retries', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })

  fireEvent.click(screen.getByTestId('mode-real')) // sim -> real, starts prepare #1
  expect(constructCount).toBe(1)

  deferreds[0].reject(new Error('download failed'))
  await waitFor(() => {
    expect(screen.getByTestId('mode-real')).toHaveAttribute('aria-pressed', 'false')
  })

  fireEvent.click(screen.getByTestId('mode-real')) // sim -> real again, guard was cleared so this retries
  expect(constructCount).toBe(2)

  deferreds[1].resolve()
  await waitFor(() => {
    expect(screen.getByTestId('btn-generate')).not.toBeDisabled()
  })
})

test('the chat example with the real model off loads it first, then runs as chat', async () => {
  render(<App />)
  fireEvent.click(screen.getAllByTestId('example-chip').at(-1)!)
  expect(constructCount).toBe(1)
  expect(runs).toHaveLength(0)
  deferreds[0].resolve()
  await waitFor(() => expect(runs).toHaveLength(1))
  expect(runs[0][0]).toBe('What is the capital of France?')
  expect(runs[0][3]).toMatchObject({ kind: 'chat' })
})

test('if loading fails, the chat example runs nothing', async () => {
  render(<App />)
  fireEvent.click(screen.getAllByTestId('example-chip').at(-1)!)
  deferreds[0].reject(new Error('download failed'))
  await waitFor(() => expect(screen.getByTestId('model-error')).toBeInTheDocument())
  expect(runs).toHaveLength(0)
})

test('the chat example after the real model was loaded and Simulated chosen switches back to real', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'x' } })
  fireEvent.click(screen.getByTestId('mode-real'))
  deferreds[0].resolve()
  await waitFor(() => expect(screen.getByTestId('btn-generate')).not.toBeDisabled())
  fireEvent.click(screen.getByTestId('mode-sim'))
  expect(screen.getByTestId('mode-sim')).toHaveAttribute('aria-pressed', 'true')
  fireEvent.click(screen.getAllByTestId('example-chip').at(-1)!)
  await waitFor(() => expect(runs).toHaveLength(1))
  expect(screen.getByTestId('mode-real')).toHaveAttribute('aria-pressed', 'true')
  expect(runs[0][3]).toMatchObject({ kind: 'chat' })
})

test('first visit: the banner shows; closing it is remembered', async () => {
  const { unmount } = render(<App />)
  expect(screen.getByTestId('sim-banner')).toBeInTheDocument()
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Simulated')
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect(screen.queryByTestId('sim-banner')).toBeNull()
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Simulated')
  unmount()
  render(<App />)
  expect(screen.queryByTestId('sim-banner')).toBeNull()
})

test('a returning user with the model cached starts in real mode without a click', async () => {
  localStorage.setItem('tsumugi.realModelUsed', '1')
  cached = true
  render(<App />)
  await waitFor(() => expect(constructCount).toBe(1))
  expect(screen.getByTestId('model-progress')).toHaveTextContent('from the browser’s cache')
  deferreds[0].resolve()
  await waitFor(() => expect(screen.getByTestId('mode-badge')).toHaveTextContent('Real model · wasm'))
})

test('a returning user whose cache was cleared starts simulated with the banner and downloads nothing', async () => {
  localStorage.setItem('tsumugi.realModelUsed', '1')
  localStorage.setItem('tsumugi.simBannerClosed', '1')
  cached = false
  render(<App />)
  await waitFor(() => expect(screen.getByTestId('sim-banner')).toBeInTheDocument())
  expect(constructCount).toBe(0)
})

test('the banner’s button loads the model; ready sets the remembered flag', async () => {
  render(<App />)
  fireEvent.click(screen.getByTestId('banner-load'))
  expect(constructCount).toBe(1)
  deferreds[0].resolve()
  await waitFor(() => expect(localStorage.getItem('tsumugi.realModelUsed')).toBe('1'))
})

test('switching back to Simulated while the chat example is loading runs nothing', async () => {
  render(<App />)
  fireEvent.click(screen.getAllByTestId('example-chip').at(-1)!)
  expect(constructCount).toBe(1)
  fireEvent.click(screen.getByTestId('mode-sim'))
  deferreds[0].resolve()
  await waitFor(() => expect(screen.getByTestId('btn-generate')).not.toBeDisabled())
  expect(runs).toHaveLength(0)
})

function holdCacheCheck() {
  let resolve!: (ok: boolean) => void
  const promise = new Promise<boolean>((r) => { resolve = r })
  cacheCheck = { promise, resolve }
  return cacheCheck
}

test('choosing Simulated during the start-up cache check keeps the app simulated', async () => {
  localStorage.setItem('tsumugi.realModelUsed', '1')
  const check = holdCacheCheck()
  render(<App />)
  fireEvent.click(screen.getByTestId('mode-sim'))
  check.resolve(true)
  await check.promise
  await new Promise((r) => setTimeout(r, 0))
  expect(constructCount).toBe(0)
  expect(screen.getByTestId('mode-sim')).toHaveAttribute('aria-pressed', 'true')
})

test('a download started during the start-up cache check is not labelled as from the cache', async () => {
  localStorage.setItem('tsumugi.realModelUsed', '1')
  const check = holdCacheCheck()
  render(<App />)
  fireEvent.click(screen.getByTestId('mode-real'))
  expect(constructCount).toBe(1)
  check.resolve(true)
  await check.promise
  await new Promise((r) => setTimeout(r, 0))
  expect(constructCount).toBe(1)
  expect(screen.queryByText(/from the browser’s cache/)).toBeNull()
})

test('the load request stays the same across renders and still loads the real model', async () => {
  render(<App />)
  const first = useRealModel.getState().request
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))   // re-renders App
  expect(screen.queryByTestId('sim-banner')).toBeNull()
  expect(useRealModel.getState().request).toBe(first)
  act(() => useRealModel.getState().request())
  expect(constructCount).toBe(1)
  await waitFor(() => expect(useRealModel.getState().loading).toBe(true))
})

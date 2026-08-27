import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import App from './App'
import { useTraceStore } from './trace/store'
import { initialPlayerState } from './player/reducer'
import { usePlayerStore } from './player/store'

let constructCount = 0
let deferreds: Array<{ resolve: () => void; reject: (err: Error) => void }> = []

vi.mock('./engine/tokenizer', async (importOriginal) => {
  const mod = await importOriginal<typeof import('./engine/tokenizer')>()
  return { ...mod, loadTokenizer: async () => mod.fallbackTokenizer() }
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
    run() { throw new Error('not used in this test') }
  }
  return { TransformersEngine: FakeTransformersEngine }
})

beforeEach(() => {
  constructCount = 0
  deferreds = []
  useTraceStore.getState().clear()
  usePlayerStore.setState({ ...initialPlayerState })
})

afterEach(() => cleanup())

test('rapid mode toggling mid-prepare constructs exactly one TransformersEngine', async () => {
  render(<App />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat sat' } })
  const toggle = screen.getByTestId('mode-toggle')

  // real -> sim -> real -> sim -> real, all while the first prepare() is still pending.
  fireEvent.click(toggle)
  fireEvent.click(toggle)
  fireEvent.click(toggle)
  fireEvent.click(toggle)
  fireEvent.click(toggle)

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
  const toggle = screen.getByTestId('mode-toggle')

  fireEvent.click(toggle) // sim -> real, starts prepare #1
  expect(constructCount).toBe(1)

  deferreds[0].reject(new Error('download failed'))
  await waitFor(() => {
    expect(screen.getByTestId('mode-toggle')).not.toBeChecked()
  })

  fireEvent.click(toggle) // sim -> real again, guard was cleared so this retries
  expect(constructCount).toBe(2)

  deferreds[1].resolve()
  await waitFor(() => {
    expect(screen.getByTestId('btn-generate')).not.toBeDisabled()
  })
})

import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { useTraceStore } from '../trace/store'
import { usePlaybackTicker, usePlayerStore } from './store'
import { initialPlayerState } from './reducer'

beforeEach(() => {
  vi.useFakeTimers()
  useTraceStore.getState().clear()
  usePlayerStore.setState({ ...initialPlayerState })
})

afterEach(() => {
  cleanup() // or the tickers of earlier tests keep stepping the cursor
  vi.useRealTimers()
})

test('while playing, the cursor moves on when the time of the step on screen is over, and playback ends on run-end', () => {
  useTraceStore.getState().append({ type: 'tokenize', tokens: [] })
  useTraceStore.getState().append({ type: 'run-end', reason: 'max-tokens' })
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 2 })
  usePlayerStore.getState().dispatch({ type: 'play' })

  renderHook(() => usePlaybackTicker())
  act(() => vi.advanceTimersByTime(10))  // nothing is on screen yet: the first step appears at once
  expect(usePlayerStore.getState().cursor).toBe(0)
  act(() => vi.advanceTimersByTime(800))  // tokenize stays for 900 ms
  expect(usePlayerStore.getState().cursor).toBe(0)
  act(() => vi.advanceTimersByTime(200))
  expect(usePlayerStore.getState().cursor).toBe(1)
  expect(usePlayerStore.getState().status).toBe('paused')  // finished trace → playback ends
})

test('each step stays on screen for its own time, not for that of the step after it', () => {
  // the Embeddings step is followed by a quick layer step: it must not flash by in a layer step's time
  useTraceStore.getState().append({ type: 'embed', cycle: 0, seqLen: 3, dims: 4, source: 'model' })
  useTraceStore.getState().append({ type: 'layer', cycle: 0, index: 0, total: 30 })
  useTraceStore.getState().append({ type: 'softmax', cycle: 0, temperature: 1, topK: [] })
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 3 })
  usePlayerStore.getState().dispatch({ type: 'play' })
  usePlayerStore.setState({ cursor: 0 })

  renderHook(() => usePlaybackTicker())
  act(() => vi.advanceTimersByTime(850))  // embed: 900 ms
  expect(usePlayerStore.getState().cursor).toBe(0)
  act(() => vi.advanceTimersByTime(100))
  expect(usePlayerStore.getState().cursor).toBe(1)
  act(() => vi.advanceTimersByTime(130))  // the layer step: 120 ms
  expect(usePlayerStore.getState().cursor).toBe(2)
})

test('parked at the frontier of a still-growing trace stays playing', () => {
  useTraceStore.getState().append({ type: 'tokenize', tokens: [] })
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 1 })
  usePlayerStore.getState().dispatch({ type: 'play' })

  renderHook(() => usePlaybackTicker())
  act(() => vi.advanceTimersByTime(2000))
  expect(usePlayerStore.getState().cursor).toBe(0)  // parked, engine still emitting
  expect(usePlayerStore.getState().status).toBe('playing')
})

test('paused: cursor does not move', () => {
  useTraceStore.getState().append({ type: 'tokenize', tokens: [] })
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 1 })
  renderHook(() => usePlaybackTicker())
  act(() => vi.advanceTimersByTime(5000))
  expect(usePlayerStore.getState().cursor).toBe(-1)
})

test('events that arrive during a wait do not restart it', () => {
  const layer = (index: number) => ({ type: 'layer' as const, cycle: 0, index, total: 30 })
  useTraceStore.getState().append({ type: 'tokenize', tokens: [] })
  useTraceStore.getState().append({ type: 'softmax', cycle: 0, temperature: 1, topK: [] }) // stays for 900 ms
  useTraceStore.getState().append(layer(0))
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 3 })
  usePlayerStore.getState().dispatch({ type: 'play' })
  usePlayerStore.setState({ cursor: 1 })

  const { rerender } = renderHook(() => usePlaybackTicker())
  // a slow model: more events arrive every 300 ms, each time before the 900 ms are over
  for (let i = 1; i <= 2; i++) {
    act(() => vi.advanceTimersByTime(300))
    act(() => {
      useTraceStore.getState().append(layer(i))
      usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 3 + i })
    })
    rerender()
    expect(usePlayerStore.getState().cursor).toBe(1)
  }
  act(() => vi.advanceTimersByTime(350)) // 950 ms since the wait began
  expect(usePlayerStore.getState().cursor).toBe(2)
})

test('parked at the newest event, playback goes on when the next one arrives', () => {
  useTraceStore.getState().append({ type: 'layer', cycle: 0, index: 0, total: 30 })
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 1 })
  usePlayerStore.getState().dispatch({ type: 'play' })
  usePlayerStore.setState({ cursor: 0 })
  const { rerender } = renderHook(() => usePlaybackTicker())
  act(() => vi.advanceTimersByTime(5000))
  expect(usePlayerStore.getState().cursor).toBe(0)
  act(() => {
    useTraceStore.getState().append({ type: 'layer', cycle: 0, index: 1, total: 30 })
    usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 2 })
  })
  rerender()
  act(() => vi.advanceTimersByTime(200)) // a layer step stays for 120 ms
  expect(usePlayerStore.getState().cursor).toBe(1)
})

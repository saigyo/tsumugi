import { renderHook } from '@testing-library/react'
import { act } from '@testing-library/react'
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
  vi.useRealTimers()
})

test('while playing, cursor advances one event per delay and pauses on run-end', () => {
  useTraceStore.getState().append({ type: 'tokenize', tokens: [] })
  useTraceStore.getState().append({ type: 'run-end', reason: 'max-tokens' })
  usePlayerStore.getState().dispatch({ type: 'traceGrew', length: 2 })
  usePlayerStore.getState().dispatch({ type: 'play' })

  renderHook(() => usePlaybackTicker())
  act(() => vi.advanceTimersByTime(1000))  // past tokenize delay
  expect(usePlayerStore.getState().cursor).toBe(0)
  act(() => vi.advanceTimersByTime(1000))  // past run-end delay
  expect(usePlayerStore.getState().cursor).toBe(1)
  expect(usePlayerStore.getState().status).toBe('paused')  // finished trace → playback ends
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

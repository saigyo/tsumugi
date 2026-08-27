import { afterEach, beforeEach, expect, test } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { makeFixtureTrace } from '../test/fixtures'
import { useTraceStore } from '../trace/store'
import { initialPlayerState } from '../player/reducer'
import { usePlayerStore } from '../player/store'
import { Controls } from './Controls'

beforeEach(() => {
  useTraceStore.setState({ events: makeFixtureTrace() })
  const len = useTraceStore.getState().events.length
  usePlayerStore.setState({ ...initialPlayerState, traceLength: len, status: 'paused', cursor: 3 })
})

afterEach(() => cleanup())

test('play and pause dispatch status changes', () => {
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-play'))
  expect(usePlayerStore.getState().status).toBe('playing')
  fireEvent.click(screen.getByTestId('btn-pause'))
  expect(usePlayerStore.getState().status).toBe('paused')
})

test('step buttons move the cursor', () => {
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-step-fwd'))
  expect(usePlayerStore.getState().cursor).toBe(4)
  fireEvent.click(screen.getByTestId('btn-step-back'))
  expect(usePlayerStore.getState().cursor).toBe(3)
})

test('scrubber seeks', () => {
  render(<Controls />)
  fireEvent.change(screen.getByTestId('scrubber'), { target: { value: '7' } })
  expect(usePlayerStore.getState().cursor).toBe(7)
})

test('live button jumps to frontier and plays', () => {
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-live'))
  const s = usePlayerStore.getState()
  expect(s.cursor).toBe(s.traceLength - 1)
  expect(s.status).toBe('playing')
})

test('speed select dispatches setSpeed', () => {
  render(<Controls />)
  fireEvent.change(screen.getByTestId('speed'), { target: { value: '2' } })
  expect(usePlayerStore.getState().speed).toBe(2)
})

test('cycle buttons jump between append events', () => {
  // fixture (2 cycles, 3 layers): append events at indices 9 and 17
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-cycle-fwd'))   // from cursor 3
  expect(usePlayerStore.getState().cursor).toBe(9)
  fireEvent.click(screen.getByTestId('btn-cycle-fwd'))
  expect(usePlayerStore.getState().cursor).toBe(17)
  fireEvent.click(screen.getByTestId('btn-cycle-back'))
  expect(usePlayerStore.getState().cursor).toBe(9)
})

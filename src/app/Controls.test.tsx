import { afterEach, beforeEach, expect, test } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { makeFixtureTrace } from '../test/fixtures'
import { useTraceStore } from '../trace/store'
import { initialPlayerState } from '../player/reducer'
import { usePlayerStore } from '../player/store'
import { scrubScale } from '../player/scrubScale'
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
  // the track's unit is playback time: the place of step 7 leads to step 7
  const place = scrubScale(useTraceStore.getState().events).at[7]
  fireEvent.change(screen.getByTestId('scrubber'), { target: { value: String(place) } })
  expect(usePlayerStore.getState().cursor).toBe(7)
})

test('live button jumps to frontier and plays', () => {
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-live'))
  const s = usePlayerStore.getState()
  expect(s.cursor).toBe(s.traceLength - 1)
  expect(s.status).toBe('playing')
})

test('Live looks picked while playback follows a run that is still being written, and only then', () => {
  const unfinished = makeFixtureTrace().slice(0, -1) // without the run-end event
  useTraceStore.setState({ events: unfinished })
  usePlayerStore.setState({ traceLength: unfinished.length })
  const { rerender } = render(<Controls />)
  expect(screen.getByTestId('btn-live')).not.toHaveClass('is-picked') // paused
  fireEvent.click(screen.getByTestId('btn-play'))
  rerender(<Controls />)
  // playing, with the cursor well behind the newest event: still following, so Live does not blink at each token
  expect(usePlayerStore.getState().cursor).toBeLessThan(unfinished.length - 1)
  expect(screen.getByTestId('btn-live')).toHaveClass('is-picked')
  expect(screen.getByTestId('btn-live')).toHaveAttribute('aria-pressed', 'true')
  fireEvent.click(screen.getByTestId('btn-pause'))
  rerender(<Controls />)
  expect(screen.getByTestId('btn-live')).not.toHaveClass('is-picked')
  expect(screen.getByTestId('btn-live')).toHaveAttribute('aria-pressed', 'false')
})

test('Live does not look picked when a finished run is replayed', () => {
  const { rerender } = render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-play'))
  rerender(<Controls />)
  expect(screen.getByTestId('btn-live')).not.toHaveClass('is-picked')
})

test('speed select dispatches setSpeed', () => {
  render(<Controls />)
  fireEvent.change(screen.getByTestId('speed'), { target: { value: '2' } })
  expect(usePlayerStore.getState().speed).toBe(2)
})

test('cycle buttons jump between append events', () => {
  // fixture (2 cycles, 3 layers, attention event per cycle): append events at indices 10 and 19
  render(<Controls />)
  fireEvent.click(screen.getByTestId('btn-cycle-fwd'))   // from cursor 3
  expect(usePlayerStore.getState().cursor).toBe(10)
  fireEvent.click(screen.getByTestId('btn-cycle-fwd'))
  expect(usePlayerStore.getState().cursor).toBe(19)
  fireEvent.click(screen.getByTestId('btn-cycle-back'))
  expect(usePlayerStore.getState().cursor).toBe(10)
})

test('while a run is being written the scrubber is laid out to its expected length, and marks how far it is written', () => {
  const run = makeFixtureTrace(6)
  const firstToken = run.findIndex((e) => e.type === 'append')
  const soFar = run.slice(0, firstToken + 1) // one of six tokens is there
  useTraceStore.setState({ events: soFar })
  usePlayerStore.setState({ traceLength: soFar.length, cursor: 3, status: 'playing' })
  const { rerender } = render(<Controls />)
  const scrubber = screen.getByTestId('scrubber') as HTMLInputElement
  const end = Number(scrubber.max)
  const finished = scrubScale(run).end
  expect(Math.abs(end - finished) / finished).toBeLessThan(0.02)
  const newest = scrubScale(soFar).at[soFar.length - 1]
  expect(parseFloat(scrubber.style.getPropertyValue('--written'))).toBeCloseTo((newest / end) * 100, 3)
  expect(parseFloat(scrubber.style.getPropertyValue('--written'))).toBeLessThan(25)
  const position = scrubber.style.getPropertyValue('--pos')

  // the next token arrives: the scale and the thumb stay, only the written mark moves on
  const secondToken = run.findIndex((e, i) => e.type === 'append' && i > firstToken)
  const more = run.slice(0, secondToken + 1)
  useTraceStore.setState({ events: more })
  usePlayerStore.setState({ traceLength: more.length })
  rerender(<Controls />)
  expect(Number(scrubber.max)).toBe(end)
  expect(scrubber.style.getPropertyValue('--pos')).toBe(position)
  expect(parseFloat(scrubber.style.getPropertyValue('--written'))).toBeCloseTo((scrubScale(more).at[more.length - 1] / end) * 100, 3)
})

test('a finished run ends where its last step is, and has no written mark', () => {
  usePlayerStore.setState({ cursor: useTraceStore.getState().events.length - 1 })
  render(<Controls />)
  const scrubber = screen.getByTestId('scrubber') as HTMLInputElement
  expect(scrubber.value).toBe(scrubber.max)
  expect(scrubber.style.getPropertyValue('--pos')).toBe('100%')
  expect(scrubber.style.getPropertyValue('--written')).toBe('')
})

test('dragging beyond what is written lands on the newest event', () => {
  const run = makeFixtureTrace(6)
  const soFar = run.slice(0, run.findIndex((e) => e.type === 'append') + 1)
  useTraceStore.setState({ events: soFar })
  usePlayerStore.setState({ traceLength: soFar.length, cursor: 3 })
  render(<Controls />)
  const scrubber = screen.getByTestId('scrubber') as HTMLInputElement
  fireEvent.change(scrubber, { target: { value: scrubber.max } })
  expect(usePlayerStore.getState().cursor).toBe(soFar.length - 1)
})

test('on the scrubber an arrow key moves one step, Home and End go to the ends, Page keys to the next token', () => {
  render(<Controls />)
  const scrubber = screen.getByTestId('scrubber')
  const last = useTraceStore.getState().events.length - 1
  fireEvent.keyDown(scrubber, { key: 'ArrowRight' })
  expect(usePlayerStore.getState().cursor).toBe(4)
  fireEvent.keyDown(scrubber, { key: 'ArrowLeft' })
  fireEvent.keyDown(scrubber, { key: 'ArrowDown' })
  expect(usePlayerStore.getState().cursor).toBe(2)
  fireEvent.keyDown(scrubber, { key: 'ArrowUp' })
  expect(usePlayerStore.getState().cursor).toBe(3)
  fireEvent.keyDown(scrubber, { key: 'End' })
  expect(usePlayerStore.getState().cursor).toBe(last)
  fireEvent.keyDown(scrubber, { key: 'ArrowRight' }) // stays on the last step
  expect(usePlayerStore.getState().cursor).toBe(last)
  fireEvent.keyDown(scrubber, { key: 'Home' })
  expect(usePlayerStore.getState().cursor).toBe(0)
  fireEvent.keyDown(scrubber, { key: 'PageUp' })
  expect(useTraceStore.getState().events[usePlayerStore.getState().cursor].type).toBe('append')
})

test('the scrubber says its position in steps, since its own numbers are playback time', () => {
  render(<Controls />)
  const total = useTraceStore.getState().events.length
  expect(screen.getByTestId('scrubber')).toHaveAttribute('aria-valuetext', `Step 4 of ${total}`)
})

test('the scrubber has no list of tick positions (browsers snap the thumb to them)', () => {
  render(<Controls />)
  expect(screen.getByTestId('scrubber')).not.toHaveAttribute('list')
  expect(document.querySelector('datalist')).toBeNull()
})

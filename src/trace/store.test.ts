import { beforeEach, expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { useTraceStore } from './store'

beforeEach(() => useTraceStore.getState().clear())

test('append adds events in order', () => {
  useTraceStore.getState().append({ type: 'tokenize', tokens: [{ id: 1, text: 'Hi' }] })
  useTraceStore.getState().append({ type: 'append', cycle: 0, token: { id: 2, text: ' there' } })
  const events = useTraceStore.getState().events
  expect(events).toHaveLength(2)
  expect(events[0].type).toBe('tokenize')
})

test('clear empties the trace', () => {
  useTraceStore.getState().append({ type: 'run-end', reason: 'max-tokens' })
  useTraceStore.getState().clear()
  expect(useTraceStore.getState().events).toEqual([])
})

test('load replaces the buffer wholesale', () => {
  useTraceStore.getState().clear()
  useTraceStore.getState().append({ type: 'run-end', reason: 'eos' })
  const events = makeFixtureTrace()
  useTraceStore.getState().load(events)
  expect(useTraceStore.getState().events).toEqual(events)
})

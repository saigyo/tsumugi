import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { paceOf } from './pacing'
import { scrubScale } from './scrubScale'

type Run = ReturnType<typeof makeFixtureTrace>
const appends = (events: Run) => events.flatMap((e, i) => (e.type === 'append' ? [i] : []))
const firstOf = (events: Run, type: Run[number]['type']) => events.findIndex((e) => e.type === type)
// how many places of the track lead to one event
const stretch = (events: Run, index: number) => {
  const scale = scrubScale(events)
  let count = 0
  for (let place = 0; place <= scale.end; place++) if (scale.indexAt(place) === index) count++
  return count
}

test('an empty run has no track', () => {
  const scale = scrubScale([])
  expect(scale.at).toEqual([])
  expect(scale.end).toBe(0)
})

test('a finished run starts at 0, ends at its last event, and the places only go up', () => {
  const run = makeFixtureTrace(4)
  const { at, end } = scrubScale(run)
  expect(at).toHaveLength(run.length)
  expect(at[0]).toBe(0)
  expect(end).toBe(at[at.length - 1])
  for (let i = 1; i < at.length; i++) expect(at[i]).toBeGreaterThan(at[i - 1])
})

test('the place of an event leads back to that event, across its whole stretch', () => {
  const run = makeFixtureTrace(3)
  const { at, indexAt } = scrubScale(run)
  for (let i = 0; i < run.length; i++) expect(indexAt(at[i])).toBe(i)
  // the stretch of one event ends where the next one begins
  const softmax = firstOf(run, 'softmax')
  const half = (at[softmax + 1] - at[softmax]) / 2
  expect(indexAt(at[softmax] + Math.floor(half) - 5)).toBe(softmax)
  expect(indexAt(at[softmax + 1] - Math.floor(half) + 5)).toBe(softmax + 1)
})

test('an event is as wide as the time playback gives it: a layer step is narrow, the sample is wide', () => {
  const run = makeFixtureTrace(4)
  const layer = stretch(run, firstOf(run, 'layer') + 1)
  const embed = stretch(run, firstOf(run, 'embed'))
  const logits = stretch(run, firstOf(run, 'logits'))
  const sample = stretch(run, firstOf(run, 'sample'))
  expect(embed).toBe(logits)
  expect(embed / layer).toBeCloseTo(paceOf('embed') / paceOf('layer'), 5)
  expect(sample / layer).toBeCloseTo(paceOf('sample') / paceOf('layer'), 5)
})

test('during playback the thumb moves as far as the time that passes', () => {
  // from one layer step to the next: the width of a layer step
  const run = makeFixtureTrace(4)
  const { at } = scrubScale(run)
  const layer = firstOf(run, 'layer')
  expect(at[layer + 1] - at[layer]).toBe(stretch(run, layer + 1))
})

test('while a run is being written, its end does not move from token to token', () => {
  const run = makeFixtureTrace(6) // six tokens, and the token limit is six
  const ends = appends(run).map((i) => scrubScale(run.slice(0, i + 1)).end)
  expect(new Set(ends).size).toBe(1)
  // and it is where the finished run ends, give or take its closing events
  const finished = scrubScale(run).end
  expect(Math.abs(ends[0] - finished) / finished).toBeLessThan(0.02)
})

test('the end stays put while the events of one token arrive in pieces, and no place moves back', () => {
  const run = makeFixtureTrace(6)
  const [first, second] = appends(run)
  const atFirst = scrubScale(run.slice(0, first + 1))
  for (let n = first + 2; n <= second + 1; n++) {
    const scale = scrubScale(run.slice(0, n))
    expect(scale.end).toBe(atFirst.end)
    expect(scale.at[3]).toBe(atFirst.at[3])
  }
})

test('before the first token is complete the track is already long, so the thumb does not start at the far end', () => {
  const run = makeFixtureTrace(6)
  const start = run.slice(0, 2)
  expect(start.map((e) => e.type)).toEqual(['run-start', 'tokenize'])
  const scale = scrubScale(start)
  expect(scale.at[1] / scale.end).toBeLessThan(0.05)
  const partly = scrubScale(run.slice(0, 5))
  expect(partly.at[4] / partly.end).toBeLessThan(0.1)
})

test('the end is never before what is written, and a place beyond it gives the newest event', () => {
  const run = makeFixtureTrace(3)
  for (let n = 1; n <= run.length; n++) {
    const scale = scrubScale(run.slice(0, n))
    expect(scale.end).toBeGreaterThanOrEqual(scale.at[n - 1])
    expect(scale.indexAt(scale.end)).toBeLessThanOrEqual(n - 1)
    expect(scale.indexAt(scale.end + 1000)).toBe(n - 1)
  }
  expect(scrubScale(run).indexAt(-5)).toBe(0)
})

test('a run that stops early is laid out longer until it ends', () => {
  const run = makeFixtureTrace(6)
  const third = appends(run)[2]
  const early = [...run.slice(0, third + 1), run[run.length - 1]] // three tokens, then the run ends
  expect(early[early.length - 1].type).toBe('run-end')
  const writing = scrubScale(early.slice(0, -1))
  expect(writing.end).toBeGreaterThan(writing.at[writing.at.length - 1] * 1.5)
  const ended = scrubScale(early)
  expect(ended.end).toBe(ended.at[ended.at.length - 1])
})

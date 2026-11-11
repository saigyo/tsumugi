import { expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { expectedLength } from './scrubScale'

const appends = (events: ReturnType<typeof makeFixtureTrace>) =>
  events.flatMap((e, i) => (e.type === 'append' ? [i] : []))

test('a finished run is as long as it is', () => {
  const run = makeFixtureTrace(4)
  expect(expectedLength(run)).toBe(run.length)
  expect(expectedLength([])).toBe(0)
})

test('while a run is being written, its expected length does not change from token to token', () => {
  const run = makeFixtureTrace(6) // six tokens, and the token limit is six
  const lengths = appends(run).map((i) => expectedLength(run.slice(0, i + 1)))
  expect(new Set(lengths).size).toBe(1)
  // and it is the length the run ends up with, give or take its closing events
  expect(Math.abs(lengths[0] - run.length)).toBeLessThanOrEqual(2)
})

test('the expected length stays put while the events of one token arrive in pieces', () => {
  const run = makeFixtureTrace(6)
  const [first, second] = appends(run)
  const atFirst = expectedLength(run.slice(0, first + 1))
  for (let n = first + 2; n <= second + 1; n++) expect(expectedLength(run.slice(0, n))).toBe(atFirst)
})

test('a position never moves backwards on the scale while a run that reaches its limit is written', () => {
  const run = makeFixtureTrace(6)
  const first = appends(run)[0]
  let before = 0
  for (let n = first + 1; n < run.length; n++) {
    const share = 3 / (expectedLength(run.slice(0, n)) - 1)
    if (n > first + 1) expect(share).toBeGreaterThanOrEqual(before - 1e-9)
    before = share
  }
})

test('before the first token is complete the scale is already long, so the thumb does not start at the far end', () => {
  const run = makeFixtureTrace(6)
  // only the start of the run and its tokens are there
  const start = run.slice(0, 2)
  expect(start.map((e) => e.type)).toEqual(['run-start', 'tokenize'])
  expect(expectedLength(start)).toBeGreaterThan(100)
  // part of the first token's events
  expect(expectedLength(run.slice(0, 5))).toBeGreaterThan(100)
})

test('the expected length is never shorter than what is written', () => {
  const run = makeFixtureTrace(3)
  for (let n = 1; n <= run.length; n++) expect(expectedLength(run.slice(0, n))).toBeGreaterThanOrEqual(n)
})

test('a run that stops early is expected to be longer until it ends', () => {
  const run = makeFixtureTrace(6)
  const third = appends(run)[2]
  const early = [...run.slice(0, third + 1), run[run.length - 1]] // three tokens, then the run ends
  expect(early[early.length - 1].type).toBe('run-end')
  expect(expectedLength(early.slice(0, -1))).toBeGreaterThan(early.length)
  expect(expectedLength(early)).toBe(early.length)
})

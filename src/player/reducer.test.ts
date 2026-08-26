import { expect, test } from 'vitest'
import { initialPlayerState as init, playerReducer as r } from './reducer'

const grown = r(init, { type: 'traceGrew', length: 5 })

test('initial state', () => {
  expect(init).toEqual({ cursor: -1, status: 'idle', speed: 1, followLive: true, traceLength: 0 })
})

test('play/pause toggle status; pause drops followLive', () => {
  const playing = r(grown, { type: 'play' })
  expect(playing.status).toBe('playing')
  const paused = r(playing, { type: 'pause' })
  expect(paused.status).toBe('paused')
  expect(paused.followLive).toBe(false)
})

test('manual step pauses and drops followLive; auto step does not', () => {
  const playing = r(grown, { type: 'play' })
  const manual = r(playing, { type: 'stepForward' })
  expect(manual).toMatchObject({ cursor: 0, status: 'paused', followLive: false })
  const auto = r(playing, { type: 'stepForward', auto: true })
  expect(auto).toMatchObject({ cursor: 0, status: 'playing', followLive: true })
})

test('cursor clamps at frontier and at 0', () => {
  let s = { ...grown, cursor: 4 }
  expect(r(s, { type: 'stepForward' }).cursor).toBe(4)
  s = { ...grown, cursor: 0 }
  expect(r(s, { type: 'stepBack' }).cursor).toBe(0)
})

test('seek clamps and drops followLive', () => {
  expect(r(grown, { type: 'seek', index: 99 })).toMatchObject({ cursor: 4, followLive: false })
  expect(r(grown, { type: 'seek', index: -3 }).cursor).toBe(0)
})

test('traceGrew only updates length', () => {
  const s = r({ ...grown, cursor: 2, followLive: false }, { type: 'traceGrew', length: 9 })
  expect(s).toMatchObject({ cursor: 2, traceLength: 9, followLive: false })
})

test('goLive jumps to frontier, follows, plays', () => {
  const s = r({ ...grown, cursor: 1, followLive: false, status: 'paused' as const }, { type: 'goLive' })
  expect(s).toMatchObject({ cursor: 4, followLive: true, status: 'playing' })
})

test('reset keeps speed, starts playing from -1', () => {
  const fast = r(grown, { type: 'setSpeed', speed: 4 })
  expect(r(fast, { type: 'reset' })).toEqual({ cursor: -1, status: 'playing', speed: 4, followLive: true, traceLength: 0 })
})

test('empty trace: stepping and seeking keep cursor at -1', () => {
  expect(r(init, { type: 'stepForward' }).cursor).toBe(-1)
  expect(r(init, { type: 'stepBack' }).cursor).toBe(-1)
  expect(r(init, { type: 'seek', index: 3 }).cursor).toBe(-1)
})

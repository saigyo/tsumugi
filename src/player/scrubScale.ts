import type { TraceEvent } from '../trace/types'
import { paceOf } from './pacing'

// Track units per unit of pace. Paces are tenths, so with 20 every place is a whole number, the middle of a
// step included.
const UNITS = 20
const widthOf = (type: TraceEvent['type']) => Math.round(paceOf(type) * UNITS)

// The stretch assumed for one token until the first token is complete (about a token of the real model). Only
// its size matters: with a long scale the thumb starts at the left instead of at the far end of a two-event run.
const PROVISIONAL_TOKEN = 18 * UNITS

export interface ScrubScale {
  /** The thumb's place for each event that is written, counted from 0. */
  at: number[]
  /** The place of the run's last event: where it is for a finished run, where it is expected while it is written. */
  end: number
  /** The event under a place on the track; a place beyond what is written gives the newest event. */
  indexAt(place: number): number
}

/**
 * The scrubber's track. Each event owns a stretch as wide as the time playback gives it, so a quick layer step
 * takes little room and the thumb moves evenly during playback; the thumb sits in the middle of its stretch.
 * A run that is still being written is laid out to the length it will have at its token limit, each further
 * token as wide as the last complete one, so that a new token does not push the thumb back.
 */
export function scrubScale(events: TraceEvent[]): ScrubScale {
  const n = events.length
  if (n === 0) return { at: [], end: 0, indexAt: () => -1 }
  // ends[i]: where the stretch of event i ends
  const ends: number[] = []
  let sum = 0
  for (const e of events) { sum += widthOf(e.type); ends.push(sum) }
  const upTo = (i: number) => (i < 0 ? 0 : ends[i])
  const first = widthOf(events[0].type) / 2
  const at = events.map((e, i) => ends[i] - widthOf(e.type) / 2 - first)

  let end = at[n - 1]
  const start = events.find((e) => e.type === 'run-start')
  if (events[n - 1].type !== 'run-end' && start?.type === 'run-start') {
    const appends = events.flatMap((e, i) => (e.type === 'append' ? [i] : []))
    const firstOfToken = events.findIndex((e) => 'cycle' in e)
    const before = upTo((firstOfToken < 0 ? n : firstOfToken) - 1)
    const done = appends.length
    const written = done > 0 ? upTo(appends[done - 1]) : before
    const token = done > 1 ? upTo(appends[done - 1]) - upTo(appends[done - 2])
      : done === 1 ? upTo(appends[0]) - before
      : Math.max(PROVISIONAL_TOKEN, sum - before)
    const left = Math.max(start.params.maxNewTokens - done, 0)
    const closing = widthOf('run-end')
    end = Math.max(sum, written + left * token) + closing / 2 - first
  }

  const indexAt = (place: number) => {
    const target = place + first
    let lo = 0, hi = n - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (ends[mid] >= target) hi = mid
      else lo = mid + 1
    }
    return lo
  }
  return { at, end, indexAt }
}

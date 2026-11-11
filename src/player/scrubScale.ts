import type { TraceEvent } from '../trace/types'

// Events per token assumed until the first token is complete. Only its size matters: with a long scale the
// thumb starts at the left instead of at the far end of a two-event run.
const PROVISIONAL_CYCLE = 30

/**
 * How many events the run will have once it is written. A finished run has its own length. A run that is still
 * being written is expected to reach its token limit, each further token taking as many events as the last
 * complete one. The scrubber is scaled to this, so that a new token does not push the thumb back.
 */
export function expectedLength(events: TraceEvent[]): number {
  const last = events[events.length - 1]
  if (!last || last.type === 'run-end') return events.length
  const start = events.find((e) => e.type === 'run-start')
  if (!start || start.type !== 'run-start') return events.length
  const appends = events.flatMap((e, i) => (e.type === 'append' ? [i] : []))
  const firstOfCycle = events.findIndex((e) => 'cycle' in e)
  const before = firstOfCycle < 0 ? events.length : firstOfCycle
  const done = appends.length
  const written = done > 0 ? appends[done - 1] + 1 : before
  const cycle = done > 1 ? appends[done - 1] - appends[done - 2]
    : done === 1 ? appends[0] + 1 - before
    : Math.max(PROVISIONAL_CYCLE, events.length - before)
  const left = Math.max(start.params.maxNewTokens - done, 0)
  // + 1: the event that ends the run
  return Math.max(events.length, written + left * cycle + 1)
}

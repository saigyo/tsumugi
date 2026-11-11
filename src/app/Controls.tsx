import type { CSSProperties } from 'react'
import { usePlayerStore } from '../player/store'
import { useTraceStore } from '../trace/store'
import { cycleTickIndices } from '../viz/selectors'
import { Icon } from '../ui/Icon'
import { expectedLength } from '../player/scrubScale'

export function Controls() {
  const { cursor, status, speed, traceLength, dispatch } = usePlayerStore()
  const events = useTraceStore((s) => s.events)
  const ticks = cycleTickIndices(events)
  // The scale is the run's expected full length: while the model is still writing, a new token then leaves the
  // thumb where it is instead of pushing it back. `written` marks how far the run exists so far.
  const last = Math.max(expectedLength(events), traceLength) - 1
  const share = (index: number) => (last > 0 ? `${(Math.max(index, 0) / last) * 100}%` : '0%')
  const fill = share(cursor)
  const written = last > traceLength - 1 ? share(traceLength - 1) : undefined

  // playing while the run is still being written: the view follows it as it grows. (Not "the cursor is on the last
  // event": every new event puts the cursor behind for a moment, and Live would blink at each token.)
  const following = status === 'playing' && events.length > 0 && events[events.length - 1].type !== 'run-end'

  const cycleFwd = () => {
    const next = ticks.find((i) => i > cursor)
    if (next !== undefined) dispatch({ type: 'seek', index: next })
  }
  const cycleBack = () => {
    const prev = [...ticks].reverse().find((i) => i < cursor)
    if (prev !== undefined) dispatch({ type: 'seek', index: prev })
  }

  return (
    <div className="controls">
      <button data-testid="btn-cycle-back" className="btn btn-icon" onClick={cycleBack} title="Previous token cycle" aria-label="Previous token cycle"><Icon name="first" /></button>
      <button data-testid="btn-step-back" className="btn btn-icon" onClick={() => dispatch({ type: 'stepBack' })} title="Step back" aria-label="Step back"><Icon name="back" /></button>
      {status === 'playing' ? (
        <button data-testid="btn-pause" className="btn btn-icon" onClick={() => dispatch({ type: 'pause' })} title="Pause" aria-label="Pause"><Icon name="pause" /></button>
      ) : (
        <button data-testid="btn-play" className="btn btn-icon" onClick={() => dispatch({ type: 'play' })} title="Play" aria-label="Play"><Icon name="play" /></button>
      )}
      <button data-testid="btn-step-fwd" className="btn btn-icon" onClick={() => dispatch({ type: 'stepForward' })} title="Step" aria-label="Step"><Icon name="forward" /></button>
      <button data-testid="btn-cycle-fwd" className="btn btn-icon" onClick={cycleFwd} title="Next token cycle" aria-label="Next token cycle"><Icon name="last" /></button>
      <input data-testid="scrubber" aria-label="Position in the run" type="range" min={0} max={Math.max(last, 0)}
        value={Math.max(cursor, 0)} list="cycle-ticks"
        style={{ '--pos': fill, ...(written === undefined ? {} : { '--written': written }) } as CSSProperties}
        onChange={(e) => dispatch({ type: 'seek', index: Number(e.target.value) })} />
      <datalist id="cycle-ticks">
        {ticks.map((i) => <option key={i} value={i} />)}
      </datalist>
      <select data-testid="speed" className="btn" aria-label="Playback speed" value={speed} onChange={(e) => dispatch({ type: 'setSpeed', speed: Number(e.target.value) })}>
        {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
      </select>
      <button data-testid="btn-live" className={`btn${following ? ' is-picked' : ''}`} aria-pressed={following} onClick={() => dispatch({ type: 'goLive' })} title="Jump to live"><Icon name="live" />Live</button>
    </div>
  )
}

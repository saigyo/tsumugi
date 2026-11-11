import type { CSSProperties } from 'react'
import { usePlayerStore } from '../player/store'
import { useTraceStore } from '../trace/store'
import { cycleTickIndices } from '../viz/selectors'
import { Icon } from '../ui/Icon'

export function Controls() {
  const { cursor, status, speed, traceLength, dispatch } = usePlayerStore()
  const events = useTraceStore((s) => s.events)
  const ticks = cycleTickIndices(events)
  const fill = traceLength > 1 ? `${(Math.max(cursor, 0) / (traceLength - 1)) * 100}%` : '0%'

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
      <input data-testid="scrubber" aria-label="Position in the run" type="range" min={0} max={Math.max(traceLength - 1, 0)}
        value={Math.max(cursor, 0)} list="cycle-ticks" style={{ '--pos': fill } as CSSProperties}
        onChange={(e) => dispatch({ type: 'seek', index: Number(e.target.value) })} />
      <datalist id="cycle-ticks">
        {ticks.map((i) => <option key={i} value={i} />)}
      </datalist>
      <select data-testid="speed" className="btn" aria-label="Playback speed" value={speed} onChange={(e) => dispatch({ type: 'setSpeed', speed: Number(e.target.value) })}>
        {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
      </select>
      <button data-testid="btn-live" className={`btn${following ? ' is-picked' : ''}`} onClick={() => dispatch({ type: 'goLive' })} title="Jump to live"><Icon name="live" />Live</button>
    </div>
  )
}

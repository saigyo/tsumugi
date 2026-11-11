import { useMemo, type CSSProperties, type KeyboardEvent } from 'react'
import { usePlayerStore } from '../player/store'
import { useTraceStore } from '../trace/store'
import { cycleTickIndices } from '../viz/selectors'
import { Icon } from '../ui/Icon'
import { scrubScale } from '../player/scrubScale'

export function Controls() {
  const { cursor, status, speed, traceLength, dispatch } = usePlayerStore()
  const events = useTraceStore((s) => s.events)
  const ticks = cycleTickIndices(events)
  // The track is laid out by playback time, and to the run's expected full length while the model is still
  // writing (see scrubScale). `written` marks how far the run exists so far.
  const scale = useMemo(() => scrubScale(events), [events])
  const share = (place: number) => (scale.end > 0 ? `${(place / scale.end) * 100}%` : '0%')
  const place = scale.at[Math.min(Math.max(cursor, 0), scale.at.length - 1)] ?? 0
  const newest = scale.at[scale.at.length - 1] ?? 0
  const written = scale.end > newest ? share(newest) : undefined

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

  // The track's unit is time, not steps, so the keys are handled here: one step per arrow key
  const scrubKey = (e: KeyboardEvent<HTMLInputElement>) => {
    const steps: Record<string, () => void> = {
      ArrowRight: () => dispatch({ type: 'seek', index: cursor + 1 }),
      ArrowUp: () => dispatch({ type: 'seek', index: cursor + 1 }),
      ArrowLeft: () => dispatch({ type: 'seek', index: cursor - 1 }),
      ArrowDown: () => dispatch({ type: 'seek', index: cursor - 1 }),
      PageUp: cycleFwd,
      PageDown: cycleBack,
      Home: () => dispatch({ type: 'seek', index: 0 }),
      End: () => dispatch({ type: 'seek', index: traceLength - 1 }),
    }
    const step = steps[e.key]
    if (!step) return
    e.preventDefault()
    step()
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
      {/* no list of tick positions: Firefox and Chromium snap the thumb to them, and the steps between two
          token boundaries could not be reached with the mouse */}
      <input data-testid="scrubber" aria-label="Position in the run" type="range" min={0} max={scale.end}
        value={place} aria-valuetext={traceLength > 0 ? `Step ${Math.max(cursor, 0) + 1} of ${traceLength}` : undefined}
        style={{ '--pos': share(place), ...(written === undefined ? {} : { '--written': written }) } as CSSProperties}
        onChange={(e) => dispatch({ type: 'seek', index: scale.indexAt(Number(e.target.value)) })}
        onKeyDown={scrubKey} />
      <select data-testid="speed" className="btn" aria-label="Playback speed" value={speed} onChange={(e) => dispatch({ type: 'setSpeed', speed: Number(e.target.value) })}>
        {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
      </select>
      <button data-testid="btn-live" className={`btn${following ? ' is-picked' : ''}`} aria-pressed={following} onClick={() => dispatch({ type: 'goLive' })} title="Jump to live"><Icon name="live" />Live</button>
    </div>
  )
}

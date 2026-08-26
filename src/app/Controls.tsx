import { usePlayerStore } from '../player/store'
import { useTraceStore } from '../trace/store'
import { cycleTickIndices } from '../viz/selectors'

export function Controls() {
  const { cursor, status, speed, traceLength, dispatch } = usePlayerStore()
  const events = useTraceStore((s) => s.events)
  const ticks = cycleTickIndices(events)

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
      <button data-testid="btn-cycle-back" onClick={cycleBack} title="Previous token cycle">|◀◀</button>
      <button data-testid="btn-step-back" onClick={() => dispatch({ type: 'stepBack' })} title="Step back">◀</button>
      {status === 'playing' ? (
        <button data-testid="btn-pause" onClick={() => dispatch({ type: 'pause' })} title="Pause">⏸</button>
      ) : (
        <button data-testid="btn-play" onClick={() => dispatch({ type: 'play' })} title="Play">▶</button>
      )}
      <button data-testid="btn-step-fwd" onClick={() => dispatch({ type: 'stepForward' })} title="Step">▶|</button>
      <button data-testid="btn-cycle-fwd" onClick={cycleFwd} title="Next token cycle">▶▶|</button>
      <input data-testid="scrubber" type="range" min={0} max={Math.max(traceLength - 1, 0)}
        value={Math.max(cursor, 0)} list="cycle-ticks"
        onChange={(e) => dispatch({ type: 'seek', index: Number(e.target.value) })} />
      <datalist id="cycle-ticks">
        {ticks.map((i) => <option key={i} value={i} />)}
      </datalist>
      <select data-testid="speed" value={speed} onChange={(e) => dispatch({ type: 'setSpeed', speed: Number(e.target.value) })}>
        {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
      </select>
      <button data-testid="btn-live" onClick={() => dispatch({ type: 'goLive' })} title="Jump to live">⏺ Live</button>
    </div>
  )
}

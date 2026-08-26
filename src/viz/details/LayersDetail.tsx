import type { Mode, TraceEvent } from '../../trace/types'

export function LayersDetail({ event, mode }: { event: Extract<TraceEvent, { type: 'layer' }>; mode: Mode }) {
  return (
    <div data-testid="detail-layers" className="detail">
      <h3>Transformer layers {mode === 'real' && <em>(schematic — real internals not exposed)</em>}</h3>
      <div className="layer-stack">
        {Array.from({ length: event.total }, (_, i) => (
          <div key={i} data-testid="layer-block" data-lit={String(i <= event.index)} className="layer-block">
            L{i}{mode === 'sim' && i === event.index && event.activationNorm != null && (
              <span className="layer-norm"> ‖h‖ {event.activationNorm}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

import type { Mode, TokenInfo, TraceEvent } from '../../trace/types'
import { AttentionHeatmap } from '../AttentionHeatmap'
import { ResidualStreamDiagram } from '../ResidualStreamDiagram'

export function LayersDetail({ event, mode, attention, attentionInRun, tokens, streamShape }: {
  event: Extract<TraceEvent, { type: 'layer' }>
  mode: Mode
  attention?: Extract<TraceEvent, { type: 'attention' }>
  attentionInRun?: boolean
  tokens?: TokenInfo[]
  streamShape?: { seqLen: number; dims: number }
}) {
  return (
    <div data-testid="detail-layers" className="detail">
      <h3>Transformer layers {mode === 'real' && !attentionInRun && <em>(schematic — real internals not exposed)</em>}</h3>
      {streamShape && (
        <ResidualStreamDiagram seqLen={streamShape.seqLen} dims={streamShape.dims} layers={event.total} />
      )}
      <div className="layer-stack">
        {Array.from({ length: event.total }, (_, i) => (
          <div key={i} data-testid="layer-block" data-lit={String(i <= event.index)} className="layer-block">
            L{i}{mode === 'sim' && i === event.index && event.activationNorm != null && (
              <span className="layer-norm"> ‖h‖ {event.activationNorm}</span>
            )}
          </div>
        ))}
      </div>
      {attention && tokens && <AttentionHeatmap heads={attention.heads} tokens={tokens} />}
    </div>
  )
}

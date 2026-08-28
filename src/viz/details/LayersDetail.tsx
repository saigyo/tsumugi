import { useState } from 'react'
import type { AttentionHead, Mode, TokenInfo, TraceEvent } from '../../trace/types'
import { AttentionGridExplorer } from '../AttentionGridExplorer'
import { AttentionHeatmap } from '../AttentionHeatmap'
import { ResidualStreamDiagram } from '../ResidualStreamDiagram'

export function LayersDetail({ event, mode, attention, attentionInRun, tokens, streamShape, grid, pinnedHeads, onPin, pinNote }: {
  event: Extract<TraceEvent, { type: 'layer' }>
  mode: Mode
  attention?: Extract<TraceEvent, { type: 'attention' }>
  attentionInRun?: boolean
  tokens?: TokenInfo[]
  streamShape?: { seqLen: number; dims: number }
  grid?: Extract<TraceEvent, { type: 'attention-grid' }>
  pinnedHeads?: AttentionHead[]
  onPin?: (layer: number, head: number) => void
  pinNote?: string | null
}) {
  const [explore, setExplore] = useState(false)
  const cycleHeads = attention?.heads ?? []
  // a pinned head that is also this cycle's showcase head under the same
  // label would duplicate its chip (and its React key) — show it once
  const heads = [...cycleHeads, ...(pinnedHeads ?? []).filter((p) =>
    !cycleHeads.some((h) => h.layer === p.layer && h.head === p.head && h.label === p.label))]
  const lastPin = pinnedHeads?.at(-1)
  const focus = lastPin ? { layer: lastPin.layer, head: lastPin.head, label: lastPin.label } : undefined
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
      {heads.length > 0 && tokens && <AttentionHeatmap heads={heads} tokens={tokens} focus={focus} />}
      {pinNote && <p data-testid="pin-note" className="attn-note">{pinNote}</p>}
      {grid && (
        <button data-testid="btn-explore-heads" className="explore-toggle"
          onClick={() => setExplore((v) => !v)}>
          {explore ? 'Hide head grid' : `Explore all heads (${grid.layers * grid.heads})`}
        </button>
      )}
      {grid && explore && <AttentionGridExplorer grid={grid} onPin={onPin ?? (() => {})} />}
    </div>
  )
}

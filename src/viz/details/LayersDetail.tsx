import { useState } from 'react'
import { LoadRealButton } from '../LoadRealButton'
import type { AttentionHead, ChatSpans, Mode, TokenInfo, TraceEvent } from '../../trace/types'
import { AttentionGridExplorer } from '../AttentionGridExplorer'
import { AttentionHeatmap } from '../AttentionHeatmap'
import { ResidualStreamDiagram } from '../ResidualStreamDiagram'
import { Term } from '../../glossary/Term'
import { EditSizeCard } from '../EditSizeCard'
import { LayersNav } from '../LayersNav'
import { LayerVotesCard } from '../LayerVotesCard'
import { MlpStepsCard } from '../MlpStepsCard'
import { NeuronCard } from '../NeuronCard'
import { useMlpLayer } from '../mlpLayerSelection'
import type { CycleNeurons, CycleVotes, EditRows } from '../selectors'

const SIM_NOTE = 'Real model only: simulated runs have no MLP to measure.'
const OLD_NOTE = 'This run was recorded before MLP data was exported.'
const PENDING_NOTE = 'Votes appear once this cycle’s scores are computed.'
const NEURON_SIM_NOTE = 'Real model only: simulated runs have no neurons to show.'
const NEURON_OLD_NOTE = 'This run was recorded before neuron data existed.'
export const STEPS_SIM_NOTE = 'Real model only: simulated runs don’t compute an MLP.'
export const STEPS_OLD_NOTE = 'This run has no step-by-step data (recorded before it existed, or left out because it was damaged).'
const EDITS_PENDING_NOTE = 'Edit sizes appear once this cycle’s layers are measured.'

export function LayersDetail({ event, mode, attention, attentionInRun, tokens, streamShape, grid, pinnedHeads, onPin, pinNote, edits, votes, neurons, run, chat }: {
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
  edits?: EditRows
  votes?: CycleVotes
  neurons?: CycleNeurons
  run?: TraceEvent
  chat?: ChatSpans
}) {
  const selectLayer = useMlpLayer((s) => s.select)
  const onLayer = run ? (l: number) => selectLayer(run, l) : undefined
  const [explore, setExplore] = useState(false)
  const noteAction = mode === 'sim' ? <LoadRealButton /> : undefined
  const cycleHeads = attention?.heads ?? []
  // a pinned head that is also this cycle's showcase head under the same
  // label would duplicate its chip (and its React key) — show it once
  const heads = [...cycleHeads, ...(pinnedHeads ?? []).filter((p) =>
    !cycleHeads.some((h) => h.layer === p.layer && h.head === p.head && h.label === p.label))]
  const lastPin = pinnedHeads?.at(-1)
  const focus = lastPin ? { layer: lastPin.layer, head: lastPin.head, label: lastPin.label } : undefined
  return (
    <div data-testid="detail-layers" className="detail">
      <h3>Transformer layers <Term id="layer" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term>{' '}
        {mode === 'real' && !attentionInRun && <em>(schematic — real internals not exposed)</em>}</h3>
      <LayersNav candidates={mode === 'real' && votes?.status === 'ok' ? votes.candidates : undefined} run={run}
        showStepper={mode === 'real' && votes?.status === 'ok'} />
      <div id="layers-overview" className="layers-section">
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
      </div>
      <div id="layers-attention" className="layers-section">
      {heads.length > 0 && tokens && <AttentionHeatmap heads={heads} tokens={tokens} focus={focus} chat={chat} />}
      {pinNote && <p data-testid="pin-note" className="attn-note">{pinNote}</p>}
      {grid && (
        <button data-testid="btn-explore-heads" className="explore-toggle"
          onClick={() => setExplore((v) => !v)}>
          {explore ? 'Hide head grid' : `Explore all heads (${grid.layers * grid.heads})`}
        </button>
      )}
      {grid && explore && <AttentionGridExplorer grid={grid} onPin={onPin ?? (() => {})} />}
      </div>
      <div data-testid="mlp-cards">
        <div id="layers-edits" className="layers-section">
          <EditSizeCard noteAction={noteAction} onLayer={onLayer} chat={chat} rows={mode === 'real' ? edits : undefined} note={mode === 'sim' ? SIM_NOTE : votes?.status === 'pending' ? EDITS_PENDING_NOTE : OLD_NOTE} />
        </div>
        <div id="layers-votes" className="layers-section">
          <LayerVotesCard
          noteAction={noteAction}
          onLayer={onLayer}
          candidates={mode === 'real' && votes?.status === 'ok' ? votes.candidates : undefined}
          note={mode === 'sim' ? SIM_NOTE : votes?.status === 'pending' ? PENDING_NOTE : OLD_NOTE} />
        </div>
        <div id="layers-neurons" className="layers-section">
          <NeuronCard
          noteAction={noteAction}
          data={mode === 'real' && neurons?.status === 'ok' ? neurons.data : undefined}
          candidates={mode === 'real' && votes?.status === 'ok' ? votes.candidates : undefined}
          note={mode === 'sim' ? NEURON_SIM_NOTE
            : neurons?.status === 'pending' || votes?.status === 'pending' ? PENDING_NOTE : NEURON_OLD_NOTE}
          run={run} />
        </div>
        <div id="layers-steps" className="layers-section">
          <MlpStepsCard
          noteAction={noteAction}
          data={mode === 'real' && neurons?.status === 'ok' ? neurons.data : undefined}
          candidates={mode === 'real' && votes?.status === 'ok' ? votes.candidates : undefined}
          note={mode === 'sim' ? STEPS_SIM_NOTE
            : neurons?.status === 'pending' || votes?.status === 'pending' ? PENDING_NOTE : STEPS_OLD_NOTE}
          run={run} />
        </div>
      </div>
    </div>
  )
}

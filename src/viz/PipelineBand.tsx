import type { TraceEvent } from '../trace/types'
import { activeStage, eventAt, type StageId } from './selectors'

const STAGES: Array<{ id: Exclude<StageId, null>; label: string }> = [
  { id: 'tokenizer', label: 'Tokenizer' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'layers', label: 'Layers' },
  { id: 'logits', label: 'Logits' },
  { id: 'sampler', label: 'Sampler' },
]

export function PipelineBand({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const active = activeStage(eventAt(events, cursor))
  return (
    <div className="pipeline-band">
      {STAGES.map((s, i) => (
        <div key={s.id} className="stage-wrap">
          {i > 0 && <span className="stage-arrow">→</span>}
          <div data-testid="stage-card" data-stage={s.id} data-active={String(active === s.id)}
            className="stage-card">
            {s.label}
          </div>
        </div>
      ))}
      <svg className="loop-arrow" viewBox="0 0 100 20" aria-label="loop back to token stream">
        <path d="M95 15 H10 M10 15 L16 10 M10 15 L16 20" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    </div>
  )
}

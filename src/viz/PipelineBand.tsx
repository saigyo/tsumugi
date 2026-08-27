import type { TraceEvent } from '../trace/types'
import { activeStage, eventAt, latestOfType, type StageId } from './selectors'

const STAGES: Array<{ id: Exclude<StageId, null>; label: string }> = [
  { id: 'tokenizer', label: 'Tokenizer' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'layers', label: 'Layers' },
  { id: 'logits', label: 'Logits' },
  { id: 'sampler', label: 'Sampler' },
]

function summaryFor(stage: Exclude<StageId, null>, events: TraceEvent[], cursor: number): string | null {
  switch (stage) {
    case 'tokenizer': {
      const e = latestOfType(events, cursor, 'tokenize')
      return e ? `${e.tokens.length} tokens` : null
    }
    case 'embeddings': {
      const e = latestOfType(events, cursor, 'embed')
      return e ? `${e.dims} dims` : null
    }
    case 'layers': {
      const e = latestOfType(events, cursor, 'layer')
      return e ? `${e.total} layers` : null
    }
    case 'logits': {
      const e = latestOfType(events, cursor, 'logits')
      return e ? `top: ${e.topK[0]?.text ?? '—'}` : null
    }
    case 'sampler': {
      const e = latestOfType(events, cursor, 'sample')
      return e ? `chose ${e.chosen.text}` : null
    }
  }
}

export function PipelineBand({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const active = activeStage(eventAt(events, cursor))
  return (
    <div className="pipeline-band">
      {STAGES.map((s, i) => {
        const summary = summaryFor(s.id, events, cursor)
        return (
          <div key={s.id} className="stage-wrap">
            {i > 0 && <span className="stage-arrow">→</span>}
            <div data-testid="stage-card" data-stage={s.id} data-active={String(active === s.id)}
              className="stage-card">
              {s.label}
              {summary && <div data-testid="stage-summary" className="stage-summary">{summary}</div>}
            </div>
          </div>
        )
      })}
      <svg className="loop-arrow" viewBox="0 0 100 20" aria-label="loop back to token stream">
        <path d="M95 15 H10 M10 15 L16 10 M10 15 L16 20" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    </div>
  )
}

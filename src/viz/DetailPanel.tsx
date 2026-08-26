import type { Mode, TraceEvent } from '../trace/types'
import { activeStage, eventAt, latestOfType } from './selectors'
import { EmbeddingsDetail } from './details/EmbeddingsDetail'
import { LayersDetail } from './details/LayersDetail'
import { TokenizerDetail } from './details/TokenizerDetail'

export function DetailPanel({ events, cursor, mode }: { events: TraceEvent[]; cursor: number; mode: Mode }) {
  const stage = activeStage(eventAt(events, cursor))
  const empty = <div data-testid="detail-empty" className="detail">Press Generate, then step through the pipeline.</div>

  switch (stage) {
    case 'tokenizer': {
      const e = latestOfType(events, cursor, 'tokenize')
      return e ? <TokenizerDetail tokens={e.tokens} truncated={e.truncated} /> : empty
    }
    case 'embeddings': {
      const e = latestOfType(events, cursor, 'embed')
      return e ? <EmbeddingsDetail event={e} /> : empty
    }
    case 'layers': {
      const e = latestOfType(events, cursor, 'layer')
      return e ? <LayersDetail event={e} mode={mode} /> : empty
    }
    default:
      return empty  // 'logits' and 'sampler' branches added in Task 14
  }
}

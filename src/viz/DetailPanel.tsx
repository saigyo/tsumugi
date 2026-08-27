import type { Mode, TraceEvent } from '../trace/types'
import { activeStage, eventAt, latestOfType } from './selectors'
import { EmbeddingsDetail } from './details/EmbeddingsDetail'
import { LayersDetail } from './details/LayersDetail'
import { TokenizerDetail } from './details/TokenizerDetail'
import { LogitsDetail } from './details/LogitsDetail'
import { SamplerDetail } from './details/SamplerDetail'
import { RunEndDetail } from './details/RunEndDetail'
import { visibleTokens } from './selectors'

export function DetailPanel({ events, cursor, mode }: { events: TraceEvent[]; cursor: number; mode: Mode }) {
  const current = eventAt(events, cursor)
  const stage = activeStage(current)
  const empty = <div data-testid="detail-empty" className="detail">Press Generate, then step through the pipeline.</div>

  if (current?.type === 'run-end') {
    const { prompt, generated } = visibleTokens(events, cursor)
    return (
      <RunEndDetail runStart={latestOfType(events, cursor, 'run-start')} runEnd={current}
        promptTokens={prompt.length} generatedTokens={generated.length} />
    )
  }

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
    case 'logits': {
      const logits = latestOfType(events, cursor, 'logits')
      const sm = latestOfType(events, cursor, 'softmax')
      return logits ? <LogitsDetail logits={logits} softmax={sm} /> : empty
    }
    case 'sampler': {
      const sm = latestOfType(events, cursor, 'softmax')
      const sample = latestOfType(events, cursor, 'sample')
      return sm ? <SamplerDetail softmax={sm} sample={sample} /> : empty
    }
    default:
      return empty
  }
}

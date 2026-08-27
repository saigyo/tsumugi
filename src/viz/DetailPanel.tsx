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
      if (!e) return empty
      // show this cycle's attention only once its event has been reached
      const attention = latestOfType(events, cursor, 'attention')
      const inCycle = attention && attention.cycle === e.cycle ? attention : undefined
      const rows = inCycle?.heads[0]?.matrix.length ?? 0
      const { prompt, generated } = visibleTokens(events, cursor)
      const tokens = [...prompt, ...generated].slice(0, rows)
      const embed = latestOfType(events, cursor, 'embed')
      const streamShape = embed ? { seqLen: embed.seqLen, dims: embed.dims } : undefined
      return <LayersDetail event={e} mode={mode} attention={inCycle} tokens={tokens} streamShape={streamShape} />
    }
    case 'logits': {
      const logits = latestOfType(events, cursor, 'logits')
      const sm = latestOfType(events, cursor, 'softmax')
      const embed = latestOfType(events, cursor, 'embed')
      const vocabSize = latestOfType(events, cursor, 'run-start')?.vocabSize
      const readout = embed ? { dims: embed.dims, vocabSize } : undefined
      return logits ? <LogitsDetail logits={logits} softmax={sm} readout={readout} /> : empty
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

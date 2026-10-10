import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const residualStream: GlossaryEntry = {
  id: 'residual-stream',
  title: 'Residual stream',
  summary: 'The matrix of token vectors that flows through every layer, one row per token.',
  stage: 'layers',
  related: ['edit', 'layer', 'embedding'],
  body: () => (
    <>
      <p>The residual stream starts as the stacked <Term id="embedding">embeddings</Term>, one 576-number row per
        token. Each <Term id="layer">layer</Term> reads it and adds its <Term id="edit">edits</Term> back. Nothing
        ever replaces it, so the original embedding stays underneath everything the layers add.</p>
      <pre className="gl-formula">{`x₀  = embeddings                 [T × 576]
x₃₀ = x₀ + all 60 edits          same shape`}</pre>
      <p>After the last layer, only the last row is read out to score the next token (see{' '}
        <Term id="logit">logit</Term>).</p>
      <p className="gl-aside">Why “stream”: each row carries information forward from layer to layer, and any layer
        can read what earlier layers wrote into it.</p>
    </>
  ),
  inRun: (run) => run.stream
    ? <p>In this run the stream is [{run.stream.seqLen} × {run.stream.dims}] and passes through {run.stream.layers} layers.</p>
    : null,
}

import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const embedding: GlossaryEntry = {
  id: 'embedding',
  title: 'Embedding',
  summary: 'The vector a token id is turned into: one learned row of the embedding matrix E.',
  stage: 'embed',
  related: ['token', 'residual-stream', 'logit'],
  body: () => (
    <>
      <p>The embedding step is a lookup, not a computation. Each token id selects one row of a matrix E with one
        row per vocabulary entry. Stacked up, the rows form the <Term id="residual-stream">residual stream</Term> that
        enters the first layer.</p>
      <pre className="gl-formula">{`x[i] = E[id_i]        E: [49 152 × 576]
x    = [T × 576]      one row per token`}</pre>
      <Section id="learned" title="Learned, not designed">
        <p>Every row starts random and is adjusted during training so that the rest of the network predicts well.
          Nobody chose what dimension 17 means. Tokens used in similar contexts end up with rows that point in
          similar directions, which is why the Embeddings card can list meaningful neighbours.</p>
      </Section>
      <Section id="position" title="No position here">
        <p>SmolLM2 adds no position vector at this stage: “cat” gets the same row wherever it appears. Position
          enters inside <Term id="attention">attention</Term>, through rotary position embeddings (RoPE) applied to
          queries and keys.</p>
      </Section>
      <Section id="tied" title="Tied with the readout">
        <p>The same 49 152 × 576 matrix is used again at the end: the final vector is compared against every row
          to score each token (see <Term id="logit">logit</Term>). The model learns one vocabulary geometry, not
          two. In the model config this is <code>tie_word_embeddings</code>.</p>
      </Section>
    </>
  ),
  inRun: (run) => run.stream
    ? <p>Your {run.stream.seqLen} tokens became {run.stream.seqLen} rows: x [{run.stream.seqLen} × {run.stream.dims}].</p>
    : null,
}

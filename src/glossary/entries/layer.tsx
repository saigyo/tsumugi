import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const layer: GlossaryEntry = {
  id: 'layer',
  title: 'Layer',
  summary: 'One transformer block: attention followed by an MLP, each adding an edit to the residual stream.',
  stage: 'layers',
  related: ['attention', 'mlp', 'edit'],
  body: () => (
    <>
      <p>SmolLM2-135M stacks 30 layers. They all have the same shape but their own weights. Each one does two
        things in order:</p>
      <pre className="gl-formula">{`x ← x + Attention(norm(x))    mixes information between tokens
x ← x + MLP(norm(x))          works on each token on its own`}</pre>
      <p><Term id="attention">Attention</Term> lets each token gather information from earlier tokens. The{' '}
        <Term id="mlp">MLP</Term> then processes each token’s vector separately. Both add their result to the{' '}
        <Term id="residual-stream">residual stream</Term> as an <Term id="edit">edit</Term>.</p>
      <p>The <code>norm</code> is RMSNorm: it rescales each vector to a standard size before the block reads it, so
        no block has to cope with vectors that grow as edits pile up.</p>
    </>
  ),
  inRun: (run) => run.stream && run.stream.layers > 0
    ? <p>This run’s model has {run.stream.layers} layers.</p>
    : null,
}

import { Term } from '../Term'
import type { GlossaryEntry } from '../types'
import { Formula } from '../../viz/Formula'

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
      <Formula tex={String.raw`\begin{aligned} x &\leftarrow x + \operatorname{Attention}(\operatorname{norm}(x)) && \text{mixes tokens} \\ x &\leftarrow x + \operatorname{MLP}(\operatorname{norm}(x)) && \text{each token alone} \end{aligned}`} />
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

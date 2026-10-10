import { Term } from '../Term'
import type { GlossaryEntry } from '../types'
import { Formula } from '../../viz/Formula'

export const mlp: GlossaryEntry = {
  id: 'mlp',
  title: 'MLP',
  summary: 'The per-token step in each layer: widen the vector, apply a nonlinearity, narrow it back.',
  stage: 'layers',
  related: ['layer', 'edit', 'attention'],
  body: () => (
    <>
      <p>MLP stands for multi-layer perceptron, also called the feed-forward network. It works on each token’s
        vector on its own, with no mixing between positions. Its result is added to the{' '}
        <Term id="residual-stream">residual stream</Term> as an <Term id="edit">edit</Term>.</p>
      <Formula tex={String.raw`\begin{aligned} n &= \operatorname{norm}(x) && \in \mathbb{R}^{576} \\ h &= \operatorname{SiLU}(W_{\text{gate}}\, n) \odot (W_{\text{up}}\, n) && \in \mathbb{R}^{1536} \\ \text{edit} &= W_{\text{down}}\, h && \in \mathbb{R}^{576} \end{aligned}`} />
      <p>The gate and up projections widen the vector to 1536 numbers; SiLU and the element-wise product let some
        of them through and suppress others; the down projection brings the result back to 576. This variant is
        called SwiGLU.</p>
      <p>About 60% of the model’s weights sit in its 30 MLPs. A common reading: the gate and up projections
        detect patterns in the vector, and the down projection writes the information associated with them.</p>
      <p className="gl-aside">Tsumugi does not show the MLP’s internals yet.</p>
    </>
  ),
}

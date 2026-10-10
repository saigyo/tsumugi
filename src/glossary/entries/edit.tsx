import { VectorStrip } from '../../viz/VectorStrip'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'
import { Formula } from '../../viz/Formula'

const x = [0.8, -0.3, 0.1, 0.6, -0.9, 0.2, 0.4, -0.1]
const dx = [0.1, 0.4, -0.2, 0.0, 0.3, -0.1, 0.0, 0.2]
const sum = x.map((v, i) => v + dx[i])
const CELL = 14

export const edit: GlossaryEntry = {
  id: 'edit',
  title: 'Edit',
  summary: 'The vector a block adds to a token’s vector in the residual stream.',
  stage: 'layers',
  related: ['residual-stream', 'attention', 'mlp'],
  body: () => (
    <>
      <p>Each <Term id="layer">layer</Term> changes every token’s vector twice, once in{' '}
        <Term id="attention">attention</Term> and once in the <Term id="mlp">MLP</Term>. Neither block overwrites
        the vector. Each one computes a new vector of the same size and the model adds it on top. That added
        vector is the edit.</p>
      <div className="gl-figure" data-testid="gl-edit-figure">
        <VectorStrip cells={x} cellW={CELL} cellH={CELL} ariaLabel="x before the edit" />
        <span>+</span>
        <VectorStrip cells={dx} cellW={CELL} cellH={CELL} ariaLabel="the edit" />
        <span>=</span>
        <VectorStrip cells={sum} cellW={CELL} cellH={CELL} ariaLabel="x after the edit" />
      </div>
      <Formula tex={String.raw`\begin{aligned} x &\leftarrow x + \operatorname{Attention}(\operatorname{norm}(x)) \\ x &\leftarrow x + \operatorname{MLP}(\operatorname{norm}(x)) \end{aligned}`} />
      <p>Because every block only adds, the final vector is the embedding plus all 60 edits. An edit can also
        point against what is already there and partly cancel it.</p>
      <p className="gl-aside">In papers: <em>block output</em>, <em>write to the residual stream</em>, or{' '}
        <em>residual update</em>.</p>
    </>
  ),
  inRun: (run) => run.stream ? (
    <p>Each layer adds two edits of [{run.stream.seqLen} × {run.stream.dims}] to your stream, one from attention
      and one from the MLP: {run.stream.layers * 2} in total.</p>
  ) : null,
}

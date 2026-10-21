import { VectorStrip } from '../../viz/VectorStrip'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'
import { Formula } from '../../viz/Formula'
import { Section } from '../Section'
import { visibleToken } from '../../viz/tokenText'

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
      <Section id="layer-0" title="Why layer 0 is so dark">
        <p>The stream starts as a token’s embedding, which is small. The first blocks write edits many times larger
          than that, so their ratio is high. It does not mean layer 0 does the most important work, only that
          there is little to compare against yet.</p>
        <p>The first token or two of a prompt are a special case: a few middle layers write enormous edits there,
          a hundred times the stream or more. Models use these first positions as a resting place for attention,
          and these huge values are part of how they do it.</p>
      </Section>
      <Section id="quiet-middle" title="Quiet middle, loud end">
        <p>As the stream grows, each edit is compared with a larger vector, so middle layers often look quiet.
          The last layers usually write large edits again: they prepare the vector for the readout, pushing it
          towards the next token’s row of E.</p>
      </Section>
      <Section id="attn-vs-mlp" title="Attention vs MLP">
        <p>Attention edits move information between positions; MLP edits work on each position alone. Seeing which
          of the two writes more at a layer tells you whether that layer mostly gathers context or mostly
          transforms what is already there. These are typical patterns in small models, not rules.</p>
      </Section>
      <p className="gl-aside">Runs saved with an earlier version of Tsumugi can show an empty layer 11 column: its
        browser model was compressed more strongly and rounded those MLP edits to zero for every prompt token after
        the first. That came from the compression, not from SmolLM2.</p>
      <p className="gl-aside">In papers: <em>block output</em>, <em>write to the residual stream</em>, or{' '}
        <em>residual update</em>.</p>
    </>
  ),
  inRun: (run) => run.stream ? (
    <>
      <p>Each layer adds two edits of [{run.stream.seqLen} × {run.stream.dims}] to your stream, one from attention
        and one from the MLP: {run.stream.layers * 2} in total.</p>
      {run.edits && <p>The largest MLP edit for “{visibleToken(run.edits.token)}” is at layer {run.edits.layer}:{' '}
        {run.edits.ratio.toFixed(2)} times the size of the stream it is added to.</p>}
    </>
  ) : null,
}

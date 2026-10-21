import { Formula } from '../../viz/Formula'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const mlpNeuron: GlossaryEntry = {
  id: 'mlp-neuron',
  title: 'MLP neuron',
  summary: 'One of the 1,536 positions inside an MLP: how strongly it fires decides how much of its column is added to the vector.',
  stage: 'layers',
  related: ['mlp', 'votes', 'edit'],
  body: () => (
    <>
      <p>Inside each <Term id="mlp">MLP</Term>, the token’s vector is widened to 1,536 numbers and narrowed back to
        576. Each of the 1,536 positions is called a neuron. Here is how one neuron changes the vector, and how
        the card splits an MLP’s vote among them.</p>
      <ol className="gl-steps">
        <li><strong>The activation.</strong> The neuron’s number for this token, SiLU(gate) × up, is its
          activation. It can be large, near zero, or negative.</li>
        <li><strong>The column.</strong> Each neuron owns one column of the down matrix, 576 numbers. The MLP’s{' '}
          <Term id="edit">edit</Term> is the sum over all neurons of activation × column, so every neuron adds its
          column, scaled by how strongly it fired.</li>
        <li><strong>The vote.</strong> A sum of edits splits into a sum of <Term id="votes">votes</Term>, so the
          MLP’s vote for a candidate is the sum of 1,536 neuron votes: activation × (column · the candidate’s row),
          scaled by the final norm like every other piece.</li>
        <li><strong>The label.</strong> To name what a column writes, it is compared with every row of the embedding
          table (each scaled by the final norm’s weight), by cosine. The five closest rows are the tokens the
          neuron pushes toward when it fires positively; the five closest to the flipped column, when it fires
          negatively.</li>
      </ol>
      <Formula tex={String.raw`\text{vote}_i(c) = h_i \,\bigl(d_i \cdot (g \odot E_c)\bigr) \,/\, \operatorname{rms}(x)`} />
      <p>Here <Formula inline tex="h_i" /> is neuron <Formula inline tex="i" />’s activation,{' '}
        <Formula inline tex="d_i" /> its column of the down matrix, <Formula inline tex="g" /> the final norm’s
        weight and <Formula inline tex="E_c" /> the candidate’s row of E.</p>
      <p>The badge says where the candidate falls in the neuron’s full ranking of the 49,152 tokens: among the top
        0.1 %, 1 % or 10 %, or further down.</p>
      <Section id="rarely-names" title="Why the list rarely names the candidate">
        <p>A neuron’s vote is only the part of its push that points along the candidate’s row. A neuron can push
          hard toward its own tokens and still add to the candidate’s score through a small share of that push,
          and a large activation makes even a small share count. Votes are also spread out: the top few neurons
          usually carry well under half of a layer’s vote.</p>
      </Section>
      <Section id="negative" title="Negative activations">
        <p>SiLU lets gate values go slightly below zero, and the up value can be negative too, so an activation is
          often negative. Then the neuron adds its column flipped, and the tokens it pushes toward are the ones
          closest to the flipped column. The card shows the list for the sign the neuron actually fired with.
          When a neuron’s vote is negative, the card shows what it pushes away from instead.</p>
      </Section>
      <Section id="early-layers" title="Early layers">
        <p>In early layers the votes are small and the token lists look like noise. The vector is still far from
          the space the readout compares with, so a column’s closest rows of E say little about what the neuron
          does there.</p>
      </Section>
      <p className="gl-aside">In runs saved with an earlier version of Tsumugi, the browser model rounded each MLP’s
        inputs before the down matrix. The card computes the neuron votes from the unrounded activations, so for
        some layers they miss the votes card’s number by a little; the total then shows “≈”.</p>
    </>
  ),
}

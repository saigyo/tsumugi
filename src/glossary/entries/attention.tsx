import { Formula } from '../../viz/Formula'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const attention: GlossaryEntry = {
  id: 'attention',
  title: 'Attention',
  summary: 'The step where each token gathers information from earlier tokens, weighted by how relevant they are.',
  stage: 'layers',
  related: ['softmax', 'layer', 'edit', 'mlp'],
  body: () => (
    <>
      <p>Each token looks back at the tokens before it (and at itself, never at later ones), decides how relevant
        each one is, and takes a weighted average of what they offer. The result is added to the{' '}
        <Term id="residual-stream">residual stream</Term> as an <Term id="edit">edit</Term>.</p>
      <p>For one head and the token at position <Formula inline tex="i" />:</p>
      <Formula tex={String.raw`\begin{aligned}
        q_i &= W_Q \cdot n_i \quad k_j = W_K \cdot n_j \quad v_j = W_V \cdot n_j \\
        a_{ij} &= \operatorname{softmax}_{j \le i}\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right) \\
        o_i &= \textstyle\sum_{j \le i} a_{ij}\, v_j
      \end{aligned}`} />
      <Section id="symbols" title="What the symbols mean">
        <dl className="gl-symbols">
          <dt><Formula inline tex="n_i" /></dt>
          <dd>The vector of the token at position <Formula inline tex="i" />, normalised (RMSNorm) as it comes from
            the residual stream: 576 numbers.</dd>
          <dt><Formula inline tex="q_i" /></dt>
          <dd>The <em>query</em>: what token <Formula inline tex="i" /> is looking for. A learned matrix{' '}
            <Formula inline tex="W_Q" /> (64 × 576) turns its vector into 64 numbers.</dd>
          <dt><Formula inline tex="k_j" /></dt>
          <dd>The <em>key</em>: what token <Formula inline tex="j" /> offers to be found by, made the same way with{' '}
            <Formula inline tex="W_K" />. Before the comparison, queries and keys are rotated by an angle that
            depends on their position (RoPE); this is the only place position enters the model.</dd>
          <dt><Formula inline tex="v_j" /></dt>
          <dd>The <em>value</em>: what token <Formula inline tex="j" /> passes on if it is attended to, made with{' '}
            <Formula inline tex="W_V" />.</dd>
          <dt><Formula inline tex="d" /></dt>
          <dd>The length of a query or key: 64 here. Dividing by <Formula inline tex={String.raw`\sqrt{d} = 8`} />{' '}
            keeps the dot products in a range where softmax does not put all the weight on a single token.</dd>
          <dt><Formula inline tex="a_{ij}" /></dt>
          <dd>The attention weight from token <Formula inline tex="i" /> to token <Formula inline tex="j" />: the
            scores <Formula inline tex={String.raw`q_i \cdot k_j / \sqrt{d}`} /> for all{' '}
            <Formula inline tex={String.raw`j \le i`} /> go through <Term id="softmax">softmax</Term>, so the weights
            are positive and add up to 1. Unlike the softmax at the end of the model, there is no temperature and no
            top-k cut. One row of a heatmap is <Formula inline tex={String.raw`a_{i1}, \dots, a_{ii}`} />.</dd>
          <dt><Formula inline tex="o_i" /></dt>
          <dd>The head’s output for token <Formula inline tex="i" />: the values of the earlier tokens, averaged with
            the weights <Formula inline tex={String.raw`a_{ij}`} />.</dd>
          <dt><Formula inline tex="W_O" /></dt>
          <dd>The outputs of all heads are placed side by side and a learned matrix{' '}
            <Formula inline tex="W_O" /> (576 × 576) turns them into the edit.</dd>
        </dl>
      </Section>
      <p>Each layer runs 9 heads in parallel, each with its own queries; 9 × 64 = 576. To save memory, SmolLM2
        shares keys and values within groups of three heads, so a layer has 9 query matrices but only 3 key and 3
        value matrices. With 30 layers, that makes 270 heads.</p>
      <p className="gl-aside">For all positions at once, the same computation is usually written{' '}
        <Formula inline tex={String.raw`\operatorname{softmax}\!\big(Q K^{\top} / \sqrt{d}\big)\, V`} />, with the
        queries, keys and values stacked as rows of matrices.</p>
      <p>In a heatmap, a row is the token doing the looking and a column is the token looked at. Darker means more
        weight, and every row adds up to 100%.</p>
      <Section id="roles" title="Head roles Tsumugi detects">
        <ul>
          <li><strong>Attention sink</strong> puts most of its weight on the first token, a resting place when nothing
            else is relevant.</li>
          <li><strong>Previous token</strong> looks one position back.</li>
          <li><strong>Induction</strong> finds an earlier occurrence of the current token and looks at what followed
            it, which lets the model continue repeated patterns.</li>
          <li><strong>Coreference</strong> looks from a pronoun to the noun it refers to, such as “it” → “horse”.</li>
          <li><strong>Distinctive</strong> marks a head whose pattern matches none of the common templates.</li>
        </ul>
      </Section>
    </>
  ),
  inRun: (run) => run.mode === 'real'
    ? <p>This run uses the real model: the heatmaps show heads detected in its actual attention weights.</p>
    : <p>This run is simulated: the heatmaps show handcrafted patterns, not a real model’s weights.</p>,
}

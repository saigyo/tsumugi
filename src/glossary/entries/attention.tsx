import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const attention: GlossaryEntry = {
  id: 'attention',
  title: 'Attention',
  summary: 'The step where each token gathers information from earlier tokens, weighted by how relevant they are.',
  stage: 'layers',
  related: ['layer', 'edit', 'mlp'],
  body: () => (
    <>
      <p>For each position, attention computes a weight for every position up to and including itself, never a
        later one. Its output is the weighted average of those positions’ value vectors, added to the{' '}
        <Term id="residual-stream">residual stream</Term> as an <Term id="edit">edit</Term>.</p>
      <pre className="gl-formula">{`weights = softmax(q · kᵀ / √d)     one row per position, sums to 1
output  = weights · v`}</pre>
      <p>Each layer runs several heads in parallel, each with its own weights. SmolLM2-135M has 9 heads in each of
        its 30 layers: 270 in total.</p>
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

import { thousands } from '../../viz/selectors'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'
import { Formula } from '../../viz/Formula'

export const logit: GlossaryEntry = {
  id: 'logit',
  title: 'Logit',
  summary: 'The raw score the model gives each vocabulary token as the next token, where higher means more likely.',
  stage: 'logits',
  related: ['softmax', 'embedding', 'residual-stream'],
  body: () => (
    <>
      <p>After the last layer, the last row of the <Term id="residual-stream">residual stream</Term> is normalised
        and compared with every row of the embedding matrix E. Each comparison is a dot product: 576 products added
        up. The result is one score per vocabulary entry.</p>
      <Formula tex={String.raw`\begin{aligned} \text{logits} &= \operatorname{norm}(x_{\text{last}}) \cdot E^{\top} \\ &\phantom{=}\ [1 \times 576] \cdot [576 \times 49\,152] \to [49\,152] \end{aligned}`} />
      <p>Scores have no fixed range and can be negative. <Term id="softmax">Softmax</Term> turns them into
        probabilities.</p>
      <Section id="tied" title="Tied with Embeddings">
        <p>This is the Embeddings matrix read the other way: there a row was looked up by id, here every row is
          scored against the final vector. Tied weights mean the model learns one vocabulary geometry, not two.</p>
      </Section>
      <Section id="similar-rows" title="Similar rows, similar scores">
        <p>Tokens whose rows of E point the same way get nearly the same dot product, which is why near-synonyms sit
          together near the top.</p>
      </Section>
      <Section id="top-k" title="Only the top candidates are kept">
        <p>The multiplication produces a score for every vocabulary entry. The trace keeps only the top-k, the
          candidates the sampler can choose from.</p>
      </Section>
    </>
  ),
  inRun: (run) => run.vocabSize
    ? <p>{thousands(run.vocabSize)} scores were computed for each token of this run; the trace kept the top {run.params.topK}.</p>
    : null,
}

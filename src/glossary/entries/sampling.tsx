import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const sampling: GlossaryEntry = {
  id: 'sampling',
  title: 'Sampling',
  summary: 'Picking the next token from the probabilities, by a weighted random draw or by taking the top one.',
  stage: 'sample',
  related: ['softmax', 'token', 'generation-settings'],
  body: () => (
    <>
      <p>With top-k sampling, the k most likely candidates keep their <Term id="softmax">softmax</Term> shares and
        one is drawn at random, weighted by those shares. A candidate at 70% wins about 7 times in 10. That is why
        the same prompt can continue differently between runs.</p>
      <p>With T = 0 (greedy), the top candidate is always taken and the output is the same every time. What
        temperature, top-k and the token limit do is in <Term id="generation-settings">Generation settings</Term>.</p>
      <p>The chosen <Term id="token">token</Term> is appended to the text and the whole pipeline runs again for the
        next one: one cycle per token.</p>
      <p className="gl-aside">The curated examples in simulated mode follow a script, so their outcome does not
        change with temperature.</p>
    </>
  ),
  inRun: (run) => (
    <p>This run: {run.params.temperature === 0 ? 'greedy (T = 0)' : `top-k with k = ${run.params.topK}, T = ${run.params.temperature}`},
      up to {run.params.maxNewTokens} new tokens.</p>
  ),
}

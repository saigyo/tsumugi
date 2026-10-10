import { Formula } from '../../viz/Formula'
import { visibleToken } from '../../viz/tokenText'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const votes: GlossaryEntry = {
  id: 'votes',
  title: 'Per-layer votes',
  summary: 'How much each block added to or took from a candidate’s score, adding up exactly to that score.',
  stage: 'layers',
  related: ['logit', 'mlp', 'edit'],
  body: () => (
    <>
      <p>A <Term id="logit">score</Term> is the final vector, after the final norm, dotted with the candidate’s row
        of E. The final vector is a sum: the embedding plus every <Term id="edit">edit</Term>. If the norm were a
        plain linear step, the score would split into one share per part.</p>
      <p>At the last position the norm divides the whole vector by one number, its root mean square, and multiplies
        each dimension by a fixed gain. Holding that number at its real value for this position makes the norm
        linear, so the split is exact:</p>
      <Formula tex={String.raw`\begin{aligned} \text{score}(c) &= \tilde e \cdot E_c + \sum_{\ell} \tilde a_\ell \cdot E_c + \sum_{\ell} \tilde m_\ell \cdot E_c \\ \tilde v &= g \odot v \,/\, \operatorname{rms}(x) \end{aligned}`} />
      <p>Each term is one block’s vote. A positive vote raised the score; a negative one lowered it. The votes card
        shows the MLP votes per layer, and a running total that adds the embedding first, then each layer’s
        attention vote and MLP vote in order. It ends exactly on the Logits card’s score.</p>
      <p className="gl-aside">This is related to, but not the same as, the <em>logit lens</em>, which applies the
        final norm again to every partial sum. That shows what the model would predict if it stopped at a layer,
        but its steps do not add up to the final score.</p>
    </>
  ),
  inRun: (run) => run.votes ? (
    <p>At this cycle the MLP of layer {run.votes.layer} added the most to “{visibleToken(run.votes.candidate)}”: {' '}
      {run.votes.value >= 0 ? '+' : '−'}{Math.abs(run.votes.value).toFixed(2)} to its score.</p>
  ) : null,
}

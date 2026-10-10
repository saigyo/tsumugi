import { Formula } from '../../viz/Formula'
import { visibleToken } from '../../viz/tokenText'
import { VoteFigure } from '../figures/VoteFigure'
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
      <p>The Layers step hands one vector to the Logits step. It looks like a single block of 576 numbers, but it
        was built piece by piece, and that is what lets you ask how much each layer contributed to one
        candidate.</p>
      <ol className="gl-steps">
        <li><strong>The final vector is a sum.</strong> No layer replaces the vector; each block adds its{' '}
          <Term id="edit">edit</Term> on top. So the vector is the token’s embedding plus 30 attention edits plus 30
          MLP edits: 61 pieces of 576 numbers each.</li>
        <li><strong>A score compares that sum with one row of E.</strong> The candidate’s{' '}
          <Term id="logit">score</Term> is the dot product of the final vector with the candidate’s row of the
          embedding matrix.</li>
        <li><strong>The comparison splits over the sum.</strong> A dot product of a sum is the sum of the dot
          products: (a + b) · E = a · E + b · E. So the score is the embedding’s comparison with the row plus each
          edit’s comparison with it.</li>
        <li><strong>Each of those terms is that piece’s vote.</strong> It says how much that one edit pushed the
          vector toward the candidate’s row (positive) or away from it (negative). The same edit votes differently
          for every candidate, because each candidate has its own row.</li>
        <li><strong>The final norm keeps the split exact.</strong> Before the comparison, the norm divides the whole
          vector by one number, its root mean square, and multiplies each dimension by a fixed weight. Applying that
          to every piece separately gives the same result, so the votes still add up to the score exactly.</li>
      </ol>
      <VoteFigure />
      <Formula tex={String.raw`\begin{aligned} \text{score}(c) &= \tilde e \cdot E_c + \sum_{\ell} \tilde a_\ell \cdot E_c + \sum_{\ell} \tilde m_\ell \cdot E_c \\ \tilde v &= g \odot v \,/\, \operatorname{rms}(x) \end{aligned}`} />
      <p>In the formula, <Formula inline tex={String.raw`e`} /> is the embedding,{' '}
        <Formula inline tex={String.raw`a_\ell`} /> and <Formula inline tex={String.raw`m_\ell`} /> are the attention
        and MLP edits of layer <Formula inline tex={String.raw`\ell`} />, <Formula inline tex={String.raw`E_c`} /> is
        the candidate’s row of E, and the tilde marks a piece after the final norm’s scaling
        (<Formula inline tex="g" /> is its weight, <Formula inline tex={String.raw`\operatorname{rms}(x)`} /> the one
        number it divides by).</p>
      <p>The votes card shows these votes per layer and a running total: the embedding first, then each layer’s
        attention and MLP votes in order. It ends exactly on the Logits card’s score.</p>
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

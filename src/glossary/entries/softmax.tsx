import { leadRatio } from '../../viz/softmaxText'
import { visibleToken } from '../../viz/tokenText'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'
import { Formula } from '../../viz/Formula'

export const softmax: GlossaryEntry = {
  id: 'softmax',
  title: 'Softmax',
  summary: 'The function that turns scores into probabilities that add up to 100%.',
  stage: 'softmax',
  related: ['logit', 'sampling'],
  body: () => (
    <>
      <p>Softmax takes the <Term id="logit">scores</Term> of the top-k candidates, divides each by the temperature
        T, exponentiates, and divides each result by the sum of all of them.</p>
      <Formula tex={String.raw`p_i = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}`} />
      <Section id="exp" title="Why exponentiate">
        <p>Scores can be negative and differ by any amount. <code>exp</code> makes them all positive and turns a
          difference of <Formula inline tex="d" /> points into a ratio of <Formula inline tex="e^{d/T}" />. At T = 1, one point ahead means about 2.7 times the
          probability; two points about 7.4 times.</p>
      </Section>
      <Section id="temperature" title="Temperature">
        <p>Dividing every score by T before exponentiating scales every lead. T below 1 exaggerates leads (sharper,
          more predictable output); T above 1 shrinks them (flatter, more random). T = 0 skips the draw and takes
          the top score: greedy decoding.</p>
      </Section>
      <Section id="top-k" title="Top-k only">
        <p>The sum in the denominator runs over the k candidates kept by top-k, so the percentages are shares among
          them. All other vocabulary tokens get nothing and can never be drawn.</p>
      </Section>
    </>
  ),
  inRun: (run) => {
    const T = run.softmax?.temperature ?? run.params.temperature
    const top = run.softmax?.top.slice(0, 3) ?? []
    return (
      <>
        <p>{T === 0
          ? 'This run uses T = 0: no draw, the top score is taken (greedy).'
          : `At this run’s T = ${T}, 1 point ahead is ${leadRatio(1, T)}× the probability, 2 points ${leadRatio(2, T)}×.`}</p>
        {top.length > 0 && (
          <p>Current top: {top.map((c) => `${visibleToken(c.text)} ${Math.round(c.prob * 100)}%`).join(', ')}.</p>
        )}
      </>
    )
  },
}

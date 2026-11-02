import type { RunEndReason } from '../../trace/types'
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

const STOPPED: Record<RunEndReason, string> = {
  eos: 'stopped because the model picked its end token',
  'max-tokens': 'stopped at the limit',
  aborted: 'was stopped by hand',
  error: 'stopped on an error',
}

export const generationSettings: GlossaryEntry = {
  id: 'generation-settings',
  title: 'Generation settings',
  summary: 'Temperature, top-k, max new tokens and seed: how the next token is picked, and how long a run goes on.',
  stage: 'sample',
  related: ['sampling', 'softmax', 'logit'],
  body: () => (
    <>
      <p>These settings change the output without changing the model. Temperature and top-k decide how the
        next token is drawn from the candidates; max new tokens decides how often that happens; the seed fixes
        the random draws.</p>
      <Section id="temperature" title="Temperature">
        <p>The model gives every candidate a <Term id="logit">score</Term>. Temperature T divides the scores
          before <Term id="softmax">softmax</Term> turns them into percentages.</p>
        <p>Below 1, leads grow: the favourite wins more often, and the text gets safer and more repetitive. Above 1,
          leads shrink: unlikely tokens are drawn more often, and the text gets more varied and makes more
          mistakes. T changes how concentrated the odds are, never the order of the candidates.</p>
        <p>At T = 0 nothing is drawn: the top candidate is taken every time (greedy), so the same prompt always
          gives the same output. The field accepts 0 to 2; the formula is
          in <Term id="softmax" section="temperature">Softmax</Term>.</p>
      </Section>
      <Section id="top-k" title="Top-k">
        <p>Only the k highest-scoring candidates take part in the draw. All others get 0%, however high T is. This
          cuts off the long tail of odd tokens that would otherwise turn up now and then. k = 1 is greedy whatever
          T is. The field accepts 1 to 10.</p>
      </Section>
      <Section id="max-new-tokens" title="Max new tokens">
        <p>The run writes at most this many tokens, one <Term id="sampling">sampling</Term> cycle each. It stops
          earlier when the model picks its end-of-sequence token (in Chat mode <code>{'<|im_end|>'}</code>); the
          run summary says which happened. With the real model every token is one full pass through all 30
          layers, so a longer run takes longer. The field accepts 1 to 100.</p>
      </Section>
      <Section id="seed" title="Seed">
        <p>When T is above 0, each cycle draws a random number to pick among the candidates. In the real model
          that draw depends only on the seed and the cycle number. So the same prompt, settings and seed give
          the same text. An empty field means a new random seed for each run. Simulated runs need no seed: the
          prompt decides them.</p>
        <p>The run summary at the end shows the seed the run used. If a run was cut off at the limit, it offers
          "Run again with N tokens". That keeps the seed, so the new text matches the old up to the old end. The
          "Copy link" button, also on the run shelf, copies a link that fills in the prompt, settings and seed.</p>
        <p>Two limits. A different graphics card or browser can round numbers slightly differently. When two
          tokens are almost tied, that can change a pick, so a shared run usually matches but is not guaranteed
          to. And if a newer model version is published after a link was made, the link runs on the newer model
          and can give different text.</p>
      </Section>
    </>
  ),
  inRun: (run) => {
    const { temperature, topK, maxNewTokens } = run.params
    const T = temperature === 0 ? 'T = 0 (greedy)' : `T = ${temperature}`
    return (
      <p>This run: {T}, top-k {topK}, up to {maxNewTokens} new tokens{run.seed !== undefined ? `, seed ${run.seed}` : ''}.{' '}
        {run.ended ? `It wrote ${run.generated} and ${STOPPED[run.ended]}.` : `${run.generated} so far.`}</p>
    )
  },
}

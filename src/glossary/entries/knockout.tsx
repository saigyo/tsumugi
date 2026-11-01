import { visibleToken } from '../../viz/tokenText'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const knockout: GlossaryEntry = {
  id: 'knockout',
  title: 'Knockout',
  summary: 'Removing one layer’s MLP and asking the model again, to see what the answer depended on.',
  stage: 'layers',
  related: ['votes', 'mlp', 'edit', 'sampling'],
  body: () => (
    <>
      <p>The <Term id="votes">votes card</Term> tells you what a layer pushed toward a token. It does not tell you
        whether the model would have chosen that token without the layer. A knockout asks that directly.</p>
      <ol className="gl-steps">
        <li><strong>Removing a layer.</strong> The layer’s <Term id="mlp">MLP</Term> still runs, but its{' '}
          <Term id="edit">edit</Term> is multiplied by zero at every position, so nothing is added to the stream.
          The attention of that layer and every other layer is left alone.</li>
        <li><strong>Asking again.</strong> The model reads the whole text so far once more and gives a probability
          for the token that was chosen at this cycle. “Measure all 30 layers” does this once per layer: 31 passes
          including the unchanged one, about two seconds for a short text with a graphics card.</li>
        <li><strong>Direct versus total effect.</strong> A vote is the direct effect: how far the layer’s own edit
          points toward the token. A knockout is the total effect: it also includes what every later layer did with
          that edit. A layer can have a large vote and no total effect if later layers make up for it, and a small
          vote and a large total effect if later layers build on what it wrote.</li>
        <li><strong>The bars and the roles.</strong> Each bar is the change in the chosen token’s probability, in
          percentage points: red for a drop, blue for a rise. A bar is outlined when the run without the layer
          would pick a different token. Up to five chips name the layers that stand out:{' '}
          <em>decisive</em> (the top token changes; at most three are shown), <em>preparatory</em> (a small direct
          vote, but the probability drops by ten points or more), <em>compensated</em> (a large direct vote, but
          little change) and <em>dampening</em> (the probability rises by ten points or more).</li>
        <li><strong>A fair comparison.</strong> “Run with MLP N off and compare” generates a new run. It copies the
          original’s tokens up to that cycle and reuses the original’s random draws for the{' '}
          <Term id="sampling">sampling</Term>, so the first difference in the Compare view is the effect of the
          removed layer and not of chance.</li>
      </ol>
      <p className="gl-aside">Take the results as a case study, not as a property of the layer. They come from one
        prompt, and a layer that matters here may not matter for another sentence.</p>
      <p className="gl-aside">Zeroing an output the model never produces can do damage of its own. Research often
        replaces the output with its average over many prompts instead (mean ablation), which is gentler. With a model
        as small as this one, removing a single MLP often changes the top token whenever its lead is small, so many
        layers come out decisive. Removing layer 0 breaks the model altogether.</p>
      <p className="gl-aside">The outlined bar’s prediction comes from a pass without the key-value cache, while the
        run with the layer off uses it. When the top two tokens are nearly tied, the run can rarely fork differently
        from what the bar promised.</p>
    </>
  ),
  inRun: (run) => {
    const k = run.knockout
    if (!k) return null
    const pts = (d: number) => `${Math.round(d * 100)} points`
    return (
      <>
        <p>Cycle {run.cycle + 1}: “{visibleToken(k.chosen)}” had {Math.round(k.baseline * 100)}% with every layer in
          place. Removing the MLP of {k.flipped === 0 ? 'no layer' : k.flipped === 1 ? 'one layer' : `${k.flipped} of ${k.layers} layers`}{' '}
          changes the top token.</p>
        {(k.drop || k.rise) && (
          <p>{k.drop && `Largest drop: MLP ${k.drop.layer}, ${pts(k.drop.delta)}.`}
            {k.drop && k.rise && ' '}
            {k.rise && `Largest rise: MLP ${k.rise.layer}, ${pts(k.rise.delta)}.`}</p>
        )}
      </>
    )
  },
}

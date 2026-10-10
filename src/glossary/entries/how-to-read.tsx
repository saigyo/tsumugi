import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const howToRead: GlossaryEntry = {
  id: 'how-to-read',
  title: 'How to read Tsumugi',
  summary: 'A short tour of the screen, from the prompt to the run archive.',
  stage: 'general',
  related: ['token', 'residual-stream', 'modes'],
  body: () => (
    <>
      <p>Tsumugi shows what a language model does to produce each new token. The screen reads top to bottom:</p>
      <ol>
        <li><strong>Prompt and mode.</strong> Type a prompt or click an example (examples start a run at once). The
          mode switch chooses between a simulated model and the real SmolLM2-135M running in your browser
          (see <Term id="modes">modes</Term>).</li>
        <li><strong>Generate.</strong> The model produces one <Term id="token">token</Term> per cycle. Every cycle runs
          the whole pipeline once.</li>
        <li><strong>Pipeline band.</strong> Six stage cards: Tokenizer → Embeddings → Layers → Logits → Softmax →
          Sampler. The highlighted card is the step on screen; click a card to jump to that step of the current
          token. The labels on the arrows are the shapes of the data passed between stages.</li>
        <li><strong>Detail panel.</strong> Below the controls: the card for the current step, with its actual
          numbers.</li>
        <li><strong>Playback.</strong> The run is recorded and replayed at a readable pace. Pause, step forward or
          back, or jump to live.</li>
        <li><strong>Run shelf.</strong> Finished runs are kept as chips. Pin them, export them, or compare two runs
          side by side.</li>
      </ol>
      <p className="gl-aside">Underlined words open their entry in this panel. The ⓘ next to each card title opens
        the entry for that stage.</p>
    </>
  ),
  inRun: (run) => run.cycles > 0
    ? <p>You’re viewing token {run.cycle + 1} of {run.cycles} of a {run.mode === 'real' ? 'real-model' : 'simulated'} run.</p>
    : null,
}

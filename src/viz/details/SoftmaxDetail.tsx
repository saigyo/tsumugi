import type { TraceEvent } from '../../trace/types'
import { latestOfType, thousands } from '../selectors'
import { shareDerivation, softmaxExplainer } from '../softmaxText'
import { SOFTMAX_CALLOUTS } from '../../glossary/callouts'
import { Term } from '../../glossary/Term'
import { visibleToken } from '../tokenText'

const BAR_W = 200

// The second step of the readout: scores → probabilities. Shows the logits
// formula again above the softmax formula so the two-step derivation reads
// top to bottom, then the explanation and one bar per surviving candidate.
export function SoftmaxDetail({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const softmax = latestOfType(events, cursor, 'softmax')
  const logits = latestOfType(events, cursor, 'logits')
  const embed = latestOfType(events, cursor, 'embed')
  const runStart = latestOfType(events, cursor, 'run-start')
  if (!softmax) return null
  const T = softmax.temperature
  const dims = embed?.dims ?? 576
  const vocab = runStart?.vocabSize ? thousands(runStart.vocabSize) : '?'
  const shown = softmax.topK.length
  // the logits of the same cycle, by id, for the per-bar derivation
  const logitOf = new Map((logits && logits.cycle === softmax.cycle ? logits.topK : []).map((c) => [c.id, c.logit]))
  const top = Math.max(...softmax.topK.map((c) => logitOf.get(c.id) ?? Number.NEGATIVE_INFINITY))
  return (
    <div data-testid="detail-softmax" className="detail">
      <h3>Softmax (T={T}) <Term id="softmax" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term></h3>
      <p>Softmax turns the <Term id="logit">scores</Term> into probabilities that sum to 100%: the second half of the readout.</p>
      {runStart?.vocabSize && (
        <p data-testid="logits-formula" className="logits-formula">
          logits = x_last · Eᵀ&emsp;[1×{dims}] · [{dims}×{vocab}] → [{vocab}]
        </p>
      )}
      <p data-testid="softmax-formula" className="logits-formula">
        p = exp(logit / T) ÷ Σ exp(logit / T)&emsp;over the top-k: {shown} of {vocab} scores
      </p>
      <p data-testid="softmax-explainer" className="softmax-explainer">{softmaxExplainer(T, shown, vocab)}</p>
      <div className="softmax-bars">
        {softmax.topK.map((c, i) => {
          const logit = logitOf.get(c.id)
          return (
            <div key={c.id} data-testid="softmax-candidate" className="softmax-candidate">
              <span className={`token-chip hue-${i % 6}`}><span className="chip-text">{visibleToken(c.text)}</span></span>
              <svg data-testid="prob-bar" data-token={c.text} width={BAR_W} height="14">
                {logit !== undefined && <title>{shareDerivation(logit, top, T, c.prob)}</title>}
                <rect width={BAR_W * c.prob} height="14" className="bar-rect" style={{ transition: 'width .4s' }} />
              </svg>
              <span className="bar-value">{Math.round(c.prob * 100)}%</span>
              {logit !== undefined && <span className="softmax-logit">from {logit.toFixed(1)}</span>}
            </div>
          )
        })}
      </div>
      <div className="embed-callouts">
        {SOFTMAX_CALLOUTS.map((c) => (
          <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="embed-callout">ⓘ {c.label}</Term>
        ))}
      </div>
    </div>
  )
}

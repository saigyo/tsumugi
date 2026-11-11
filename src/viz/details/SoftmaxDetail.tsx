import type { TraceEvent } from '../../trace/types'
import { latestOfType } from '../selectors'
import { shareDerivation, softmaxExplainer } from '../softmaxText'
import { SOFTMAX_CALLOUTS } from '../../glossary/callouts'
import { Term } from '../../glossary/Term'
import { visibleToken } from '../tokenText'
import { Formula } from '../Formula'
import { logitsTex, softmaxTex } from '../tex'
import { Icon } from '../../ui/Icon'
import { count, fixed, percent } from '../../ui/format'

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
  const vocab = runStart?.vocabSize ? count(runStart.vocabSize) : '?'
  const shown = softmax.topK.length
  // the logits of the same cycle, by id, for the per-bar derivation
  const logitOf = new Map((logits && logits.cycle === softmax.cycle ? logits.topK : []).map((c) => [c.id, c.logit]))
  const top = Math.max(...softmax.topK.map((c) => logitOf.get(c.id) ?? Number.NEGATIVE_INFINITY))
  return (
    <div data-testid="detail-softmax" className="detail">
      <h3>Softmax (T={T}) <Term id="softmax" variant="icon" label="About this stage" testId="stage-info"><Icon name="info" /></Term></h3>
      <p>Softmax turns the <Term id="logit">scores</Term> into probabilities that sum to 100%: the second half of the readout.</p>
      {runStart?.vocabSize && (
        <Formula testId="logits-formula" className="card-formula" inline tex={logitsTex(dims, runStart.vocabSize)} />
      )}
      <Formula testId="softmax-formula" className="card-formula" inline tex={softmaxTex(shown, runStart?.vocabSize)} />
      <p data-testid="softmax-explainer" className="softmax-explainer">{softmaxExplainer(T, shown, vocab)}</p>
      <div className="softmax-bars">
        {softmax.topK.map((c, i) => {
          const logit = logitOf.get(c.id)
          return (
            <div key={c.id} data-testid="softmax-candidate" data-top={String(i === 0)} className="softmax-candidate">
              <span className={`tok${i === 0 ? ' is-chosen' : ''}`} title={visibleToken(c)}><span className="chip-text">{visibleToken(c)}</span></span>
              <svg data-testid="prob-bar" data-token={c.text} className="softmax-bar" width="100%" height="14" viewBox={`0 0 ${BAR_W} 14`} preserveAspectRatio="none">
                {logit !== undefined && <title>{shareDerivation(logit, top, T, c.prob)}</title>}
                <rect y="2" width={BAR_W * c.prob} height="10" rx="2" className={`bar-svg${i === 0 ? ' bar-svg-chosen' : ''}`} style={{ transition: 'width .4s' }} />
              </svg>
              <span className="bar-value">{percent(c.prob)}</span>
              {logit !== undefined && <span className="softmax-logit">from {fixed(logit, 1)}</span>}
            </div>
          )
        })}
      </div>
      <div className="embed-callouts">
        {SOFTMAX_CALLOUTS.map((c) => (
          <Term key={c.label} id={c.id} section={c.section} tip={c.tip} variant="chip" testId="embed-callout"><Icon name="info" /> {c.label}</Term>
        ))}
      </div>
    </div>
  )
}

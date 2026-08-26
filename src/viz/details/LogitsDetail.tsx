import type { TraceEvent } from '../../trace/types'

type LogitsEvent = Extract<TraceEvent, { type: 'logits' }>
type SoftmaxEvent = Extract<TraceEvent, { type: 'softmax' }>

export function LogitsDetail({ logits, softmax }: { logits: LogitsEvent; softmax?: SoftmaxEvent }) {
  const showProbs = softmax != null && softmax.cycle === logits.cycle
  const maxLogit = Math.max(...logits.topK.map((c) => c.logit))
  return (
    <div data-testid="detail-logits" className="detail">
      <h3>{showProbs ? `Softmax (T=${softmax!.temperature})` : 'Logits'}</h3>
      <p>{showProbs
        ? 'Softmax turns scores into probabilities that sum to 100%.'
        : 'The model scores every vocabulary token; these are the top candidates.'}</p>
      <div className="bar-chart">
        {logits.topK.map((c, i) => {
          const frac = showProbs ? softmax!.topK[i].prob : Math.max(0.02, c.logit / maxLogit)
          return (
            <div key={c.id} className="bar-row">
              <span className="bar-label">{c.text}</span>
              <svg data-testid="logit-bar" data-token={c.text} width="200" height="14">
                <rect width={200 * frac} height="14" className="bar-rect"
                  style={{ transition: 'width .4s' }} />
              </svg>
              <span className="bar-value">
                {showProbs ? `${Math.round(softmax!.topK[i].prob * 100)}%` : c.logit.toFixed(1)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

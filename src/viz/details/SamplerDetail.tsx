import type { TraceEvent } from '../../trace/types'
import { Term } from '../../glossary/Term'
import { visibleToken } from '../tokenText'
import { Icon } from '../../ui/Icon'

type SoftmaxEvent = Extract<TraceEvent, { type: 'softmax' }>
type SampleEvent = Extract<TraceEvent, { type: 'sample' }>

export function SamplerDetail({ softmax, sample }: { softmax: SoftmaxEvent; sample?: SampleEvent }) {
  let offset = 0
  return (
    <div data-testid="detail-sampler" className="detail">
      <h3>Sampler{sample && ` (${sample.method})`} <Term id="sampling" variant="icon" label="About this stage" testId="stage-info"><Icon name="info" /></Term></h3>
      {sample?.method === 'forced'
        ? <p>Copied from the original run (knockout replay): no <Term id="sampling">draw</Term>.</p>
        : <p>A weighted <Term id="sampling">draw</Term> across the probability strip picks the next token.</p>}
      <svg width="400" height="30" className="roulette">
        {softmax.topK.map((c) => {
          const x = offset
          const w = 400 * c.prob
          offset += w
          const chosen = sample?.chosen.id === c.id
          return <rect key={c.id} x={x} width={Math.max(w, 1)} height="30"
            data-chosen={String(chosen)} className={chosen ? 'slice slice-chosen' : 'slice'} />
        })}
      </svg>
      {sample && (
        <p data-testid="chosen-marker" className="chosen-marker">
          → chose <span className="tok is-chosen"><span className="chip-text">{visibleToken(sample.chosen)}</span></span> (id {sample.chosen.id})
        </p>
      )}
    </div>
  )
}

import type { TraceEvent } from '../../trace/types'

type SoftmaxEvent = Extract<TraceEvent, { type: 'softmax' }>
type SampleEvent = Extract<TraceEvent, { type: 'sample' }>

export function SamplerDetail({ softmax, sample }: { softmax: SoftmaxEvent; sample?: SampleEvent }) {
  let offset = 0
  return (
    <div data-testid="detail-sampler" className="detail">
      <h3>Sampling {sample && `(${sample.method})`}</h3>
      <p>A weighted draw across the probability strip picks the next token.</p>
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
          → chose "{sample.chosen.text}" (id {sample.chosen.id})
        </p>
      )}
    </div>
  )
}

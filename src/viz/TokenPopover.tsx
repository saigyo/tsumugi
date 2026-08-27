import type { TraceEvent } from '../trace/types'

type SoftmaxEvent = Extract<TraceEvent, { type: 'softmax' }>
type SampleEvent = Extract<TraceEvent, { type: 'sample' }>

export function TokenPopover({ softmax, sample }: { softmax: SoftmaxEvent; sample: SampleEvent }) {
  return (
    <div data-testid="token-popover" className="token-popover" role="tooltip">
      <div className="popover-title">sampled from (T={softmax.temperature})</div>
      {softmax.topK.map((c) => (
        <div key={c.id} data-testid="popover-row" data-chosen={String(c.id === sample.chosen.id)}
          className="popover-row">
          <span className="popover-token">{c.text}</span>
          <span className="popover-bar" style={{ width: `${Math.max(2, c.prob * 100)}px` }} />
          <span className="popover-pct">{Math.round(c.prob * 100)}%</span>
        </div>
      ))}
    </div>
  )
}

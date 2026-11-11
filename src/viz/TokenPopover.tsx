import type { TraceEvent } from '../trace/types'
import { visibleToken } from './tokenText'
import { percent } from '../ui/format'

type SoftmaxEvent = Extract<TraceEvent, { type: 'softmax' }>
type SampleEvent = Extract<TraceEvent, { type: 'sample' }>

export function TokenPopover({ softmax, sample, note }: { softmax: SoftmaxEvent; sample: SampleEvent; note?: string }) {
  return (
    <div data-testid="token-popover" className="token-popover pop" role="tooltip">
      <div className="popover-title">sampled from (T={softmax.temperature})</div>
      {note && <div data-testid="popover-note" className="popover-note">{note}</div>}
      {softmax.topK.map((c) => (
        <div key={c.id} data-testid="popover-row" data-chosen={String(c.id === sample.chosen.id)}
          className="popover-row">
          <span className="popover-token"><span className={`tok tok-sm${c.id === sample.chosen.id ? ' is-chosen' : ''}`}>{visibleToken(c)}</span></span>
          <span className={`bar${c.id === sample.chosen.id ? ' bar-chosen' : ''}`} style={{ width: `${Math.max(2, c.prob * 100)}px` }} />
          <span className="popover-pct">{percent(c.prob)}</span>
        </div>
      ))}
    </div>
  )
}

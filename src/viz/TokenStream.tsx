import { useState } from 'react'
import type { TraceEvent } from '../trace/types'
import { distributionFor, visibleTokens } from './selectors'
import { TokenPopover } from './TokenPopover'

export function TokenStream({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const { prompt, generated } = visibleTokens(events, cursor)
  const [hoveredCycle, setHoveredCycle] = useState<number | null>(null)
  const dist = hoveredCycle !== null ? distributionFor(events, hoveredCycle) : undefined

  return (
    <div className="token-stream" aria-label="Token stream">
      {prompt.map((t, i) => (
        <span key={`p${i}`} data-testid="prompt-token" className="token token-prompt" title={`id ${t.id}`}>
          {t.text}
        </span>
      ))}
      {generated.map((t, i) => (
        <span key={`g${i}`} data-testid="generated-token" className="token token-generated" title={`id ${t.id}`}
          onMouseEnter={() => setHoveredCycle(t.cycle)}
          onMouseLeave={() => setHoveredCycle(null)}>
          {t.text}
          {hoveredCycle === t.cycle && dist && <TokenPopover softmax={dist.softmax} sample={dist.sample} />}
        </span>
      ))}
    </div>
  )
}

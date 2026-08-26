import type { TraceEvent } from '../trace/types'
import { visibleTokens } from './selectors'

export function TokenStream({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const { prompt, generated } = visibleTokens(events, cursor)
  return (
    <div className="token-stream" aria-label="Token stream">
      {prompt.map((t, i) => (
        <span key={`p${i}`} data-testid="prompt-token" className="token token-prompt" title={`id ${t.id}`}>
          {t.text}
        </span>
      ))}
      {generated.map((t, i) => (
        <span key={`g${i}`} data-testid="generated-token" className="token token-generated" title={`id ${t.id}`}>
          {t.text}
        </span>
      ))}
    </div>
  )
}

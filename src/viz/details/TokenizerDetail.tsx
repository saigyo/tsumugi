import type { TokenInfo } from '../../trace/types'
import { SPACE_MARKER, visibleToken } from '../tokenText'

export function TokenizerDetail({ tokens, truncated }: { tokens: TokenInfo[]; truncated?: boolean }) {
  const anySpace = tokens.some((t) => t.text.startsWith(' '))
  return (
    <div data-testid="detail-tokenizer" className="detail">
      <h3>Tokenizer</h3>
      <p>
        The prompt is split into {tokens.length} tokens, each mapped to a vocabulary ID.
        {anySpace && (
          <span data-testid="tokenizer-space-note">
            {' '}{SPACE_MARKER} marks a leading space — it belongs to the token, so “{SPACE_MARKER}cat” and “cat” are different vocabulary entries.
          </span>
        )}
      </p>
      {truncated && (
        <p data-testid="truncation-notice" className="notice">
          Prompt was longer than the model's context window — it was truncated to fit.
        </p>
      )}
      <div className="token-chip-row">
        {tokens.map((t, i) => (
          <span key={i} className={`token-chip hue-${i % 6}`}>
            <span className="chip-text">{visibleToken(t.text)}</span>
            <span className="chip-id">{t.id}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

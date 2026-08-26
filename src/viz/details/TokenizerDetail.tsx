import type { TokenInfo } from '../../trace/types'

export function TokenizerDetail({ tokens, truncated }: { tokens: TokenInfo[]; truncated?: boolean }) {
  return (
    <div data-testid="detail-tokenizer" className="detail">
      <h3>Tokenizer</h3>
      <p>The prompt is split into {tokens.length} tokens, each mapped to a vocabulary ID.</p>
      {truncated && (
        <p data-testid="truncation-notice" className="notice">
          Prompt was longer than the model's context window — it was truncated to fit.
        </p>
      )}
      <div className="token-chip-row">
        {tokens.map((t, i) => (
          <span key={i} className={`token-chip hue-${i % 6}`}>
            <span className="chip-text">{t.text}</span>
            <span className="chip-id">{t.id}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

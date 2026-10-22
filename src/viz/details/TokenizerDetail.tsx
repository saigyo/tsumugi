import { useState } from 'react'
import type { ChatSpans, TokenInfo } from '../../trace/types'
import { SPACE_MARKER, visibleToken } from '../tokenText'
import { Term } from '../../glossary/Term'
import { FOLD_TITLE, foldItems, foldLabel, tokenRole, type TokenRole } from '../chatFold'

const ROLE_CLASS: Record<TokenRole, string> = { prompt: '', template: ' token-template', special: ' token-special' }

export function TokenizerDetail({ tokens, truncated, chat }: { tokens: TokenInfo[]; truncated?: boolean; chat?: ChatSpans }) {
  // the fold is opened for one token list: a new run closes it again
  const [openFor, setOpenFor] = useState<TokenInfo[] | null>(null)
  const open = openFor === tokens
  const anySpace = tokens.some((t) => t.text.startsWith(' '))
  return (
    <div data-testid="detail-tokenizer" className="detail">
      <h3>Tokenizer <Term id="token" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term></h3>
      <p>
        The prompt is split into {tokens.length} <Term id="token">tokens</Term>, each mapped to a vocabulary ID.
        {anySpace && (
          <span data-testid="tokenizer-space-note">
            {' '}{SPACE_MARKER} marks a leading space — it belongs to the token, so “{SPACE_MARKER}cat” and “cat” are different vocabulary entries.
          </span>
        )}
      </p>
      {chat && (
        <p data-testid="tokenizer-chat-note">
          In Chat mode the prompt is wrapped in the model's <Term id="chat-template" section="what">chat template</Term>:{' '}
          {tokens.length - (chat.user[1] - chat.user[0])} of the {tokens.length} tokens were added by the template.
        </p>
      )}
      {truncated && (
        <p data-testid="truncation-notice" className="notice">
          Prompt was longer than the model's context window — it was truncated to fit.
        </p>
      )}
      <div className="token-chip-row">
        {foldItems(tokens.length, chat?.system, open).map((item) => item.kind === 'fold' ? (
          <button key="fold" type="button" data-testid="system-fold" className="token-chip token-fold"
            aria-expanded={open} title={FOLD_TITLE} onClick={() => setOpenFor(open ? null : tokens)}>
            {foldLabel(item.count, open)}
          </button>
        ) : (
          <span key={item.index}
            className={`token-chip hue-${item.index % 6}${ROLE_CLASS[tokenRole(item.index, tokens[item.index].text, chat)]}`}>
            <span className="chip-text">{visibleToken(tokens[item.index])}</span>
            <span className="chip-id">{tokens[item.index].id}</span>
          </span>
        ))}
      </div>
    </div>
  )
}

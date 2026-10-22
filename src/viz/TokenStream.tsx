import { useState } from 'react'
import type { TraceEvent } from '../trace/types'
import { chatLayout, distributionFor, visibleTokens } from './selectors'
import { TokenPopover } from './TokenPopover'
import { visibleToken } from './tokenText'
import { FOLD_TITLE, foldItems, foldLabel, tokenRole } from './chatFold'

const ROLE_CLASS = { prompt: '', template: ' token-template', special: ' token-special' } as const
const END_TOKEN = '<|im_end|>'

export function TokenStream({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const { prompt, generated } = visibleTokens(events, cursor)
  const chat = chatLayout(events)
  // the fold is opened for one run: a new run-start event closes it again
  const [openFor, setOpenFor] = useState<TraceEvent | null>(null)
  const open = events[0] !== undefined && openFor === events[0]
  const [hoveredCycle, setHoveredCycle] = useState<number | null>(null)
  const dist = hoveredCycle !== null ? distributionFor(events, hoveredCycle) : undefined

  return (
    <div className="token-stream" aria-label="Token stream">
      {foldItems(prompt.length, chat?.system, open).map((item) => item.kind === 'fold' ? (
        <button key="fold" type="button" data-testid="system-fold" className="token token-fold"
          aria-expanded={open} title={FOLD_TITLE} onClick={() => setOpenFor(open ? null : events[0])}>
          {foldLabel(item.count, open)}
        </button>
      ) : (
        <span key={`p${item.index}`} data-testid="prompt-token"
          className={`token token-prompt${ROLE_CLASS[tokenRole(item.index, prompt[item.index].text, chat)]}`}
          title={`id ${prompt[item.index].id}`}>
          {visibleToken(prompt[item.index])}
        </span>
      ))}
      {generated.map((t, i) => {
        const end = chat !== undefined && t.text === END_TOKEN
        return (
          <span key={`g${i}`} data-testid="generated-token"
            className={`token token-generated${end ? ' token-end' : ''}`}
            title={end ? `id ${t.id} · end of turn: the model stopped here` : `id ${t.id}`}
            onMouseEnter={() => setHoveredCycle(t.cycle)}
            onMouseLeave={() => setHoveredCycle(null)}>
            {visibleToken(t)}
            {hoveredCycle === t.cycle && dist && <TokenPopover softmax={dist.softmax} sample={dist.sample} />}
          </span>
        )
      })}
    </div>
  )
}

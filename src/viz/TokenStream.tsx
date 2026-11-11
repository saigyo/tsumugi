import { useState } from 'react'
import type { TraceEvent } from '../trace/types'
import { chatLayout, distributionFor, visibleTokens } from './selectors'
import { TokenPopover } from './TokenPopover'
import { visibleToken } from './tokenText'
import { FOLD_TITLE, foldItems, foldLabel, tokenRole } from './chatFold'
import { Icon } from '../ui/Icon'

const ROLE_CLASS = { prompt: '', template: ' token-template', special: ' token-special' } as const
const END_TOKEN = '<|im_end|>'

export function TokenStream({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const { prompt, generated } = visibleTokens(events, cursor)
  const chat = chatLayout(events)
  // the fold is opened for one run: a new run-start event closes it again
  const [openFor, setOpenFor] = useState<TraceEvent | null>(null)
  const open = events[0] !== undefined && openFor === events[0]
  const [hoveredCycle, setHoveredCycle] = useState<number | null>(null)
  // the newest token is the one the next prediction continues from
  const current = generated.length > 0 ? `g${generated.length - 1}` : prompt.length > 0 ? `p${prompt.length - 1}` : null
  const dist = hoveredCycle !== null ? distributionFor(events, hoveredCycle) : undefined

  return (
    <div data-testid="token-stream" className="token-stream" aria-label="Token stream">
      {foldItems(prompt.length, chat?.system, open).map((item) => item.kind === 'fold' ? (
        <button key="fold" type="button" data-testid="system-fold" className="token token-fold"
          aria-expanded={open} title={FOLD_TITLE} onClick={() => setOpenFor(open ? null : events[0])}>
          {foldLabel(item.count)} <Icon name="chevron" />
        </button>
      ) : (
        <span key={`p${item.index}`} data-testid="prompt-token"
          className={`token token-prompt${ROLE_CLASS[tokenRole(item.index, prompt[item.index].text, chat)]}${current === `p${item.index}` ? ' token-current' : ''}`}
          title={`id ${prompt[item.index].id}`}>
          {visibleToken(prompt[item.index])}
        </span>
      ))}
      {generated.map((t, i) => {
        const end = chat !== undefined && t.text === END_TOKEN
        const note = end ? `id ${t.id} · end of turn: the model stopped here` : `id ${t.id}`
        // the popover carries the id while it shows: a native tooltip would sit on top of it
        const popped = hoveredCycle === t.cycle && dist !== undefined && dist !== null
        return (
          <span key={`g${i}`} data-testid="generated-token"
            className={`token token-generated${end ? ' token-end' : current === `g${i}` ? ' token-current' : ''}`}
            title={popped ? undefined : note}
            onMouseEnter={() => setHoveredCycle(t.cycle)}
            onMouseLeave={() => setHoveredCycle(null)}>
            {visibleToken(t)}
            {popped && <TokenPopover softmax={dist.softmax} sample={dist.sample} note={note} />}
          </span>
        )
      })}
    </div>
  )
}

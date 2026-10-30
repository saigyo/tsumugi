import { create } from 'zustand'
import type { ChatSpans } from '../trace/types'

// Whether the Layers detail shows a chat run's system prompt in full: the
// matrix's "Show system prompt" box and the Edit sizes fold. Kept here, not in
// the cards, because scrubbing unmounts them (the matrix during a cycle's layer
// steps, the whole Layers detail from Logits to Sampler). Tied to the run's
// chat spans (its tokenize event's object): a new run, another archived run or
// an import starts folded again.
export type ChatFold = 'matrix' | 'edits'

interface ChatFolds {
  open: { spans: ChatSpans; folds: ChatFold[] } | null
  set: (spans: ChatSpans, fold: ChatFold, on: boolean) => void
}

export const useChatFolds = create<ChatFolds>((set) => ({
  open: null,
  set: (spans, fold, on) => set((s) => {
    const folds = s.open?.spans === spans ? s.open.folds.filter((f) => f !== fold) : []
    return { open: { spans, folds: on ? [...folds, fold] : folds } }
  }),
}))

export const resetChatFolds = () => useChatFolds.setState({ open: null })

export function useChatFold(fold: ChatFold, spans: ChatSpans | undefined): [boolean, (on: boolean) => void] {
  const open = useChatFolds((s) => spans !== undefined && s.open?.spans === spans && s.open.folds.includes(fold))
  const set = useChatFolds((s) => s.set)
  return [open, (on) => { if (spans) set(spans, fold, on) }]
}

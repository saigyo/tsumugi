import type { ChatSpans } from '../trace/types'
import { isSpecialToken } from '../engine/chatTemplate'

// Folding a chat run's system turn: one item per shown token, plus one fold
// item where the system turn starts. Open, the folded tokens follow the item.
export type FoldItem = { kind: 'token'; index: number } | { kind: 'fold'; count: number }

export function foldItems(length: number, system: [number, number] | undefined, open: boolean): FoldItem[] {
  const all: FoldItem[] = Array.from({ length }, (_, index) => ({ kind: 'token', index }))
  if (!system || system[0] >= system[1] || system[0] >= length) return all
  const fold: FoldItem = { kind: 'fold', count: system[1] - system[0] }
  return [...all.slice(0, system[0]), fold, ...all.slice(open ? system[0] : system[1])]
}

export type TokenRole = 'prompt' | 'template' | 'special'

// A prompt position's role in a chat run; every position of a plain run is 'prompt'.
export function tokenRole(index: number, text: string, chat: ChatSpans | undefined): TokenRole {
  if (!chat) return 'prompt'
  if (isSpecialToken(text)) return 'special'
  return index >= chat.user[0] && index < chat.user[1] ? 'prompt' : 'template'
}

export const foldLabel = (count: number, open: boolean): string =>
  `system prompt · ${count} tokens ${open ? '▾' : '▸'}`

export const FOLD_TITLE = 'The system turn the chat template adds before your message. See “Chat template” in the glossary.'

import { readableTokenText } from '../engine/tokenBytes'

// Byte-level BPE tokens carry their whitespace: " cat" and "cat" are
// different vocabulary entries, and "\n", "\t" or a run of spaces are tokens
// of their own. Rendered raw they are invisible, so the UI swaps them for
// glyphs: a leading run of spaces becomes one ␣ per space, newlines ↵,
// tabs ⇥ — the rest of the text is left untouched. Passed whole (id and
// text), a token that is only part of a UTF-8 character shows its bytes
// (<0xE2>) instead of "�" (tokenBytes.ts).
export const SPACE_MARKER = '␣'
export const NEWLINE_MARKER = '↵'
export const TAB_MARKER = '⇥'

export function visibleToken(token: string | { id: number; text: string }): string {
  const text = typeof token === 'string' ? token : readableTokenText(token.id, token.text)
  const leading = text.match(/^ +/)?.[0].length ?? 0
  return SPACE_MARKER.repeat(leading)
    + text.slice(leading).replace(/\r?\n/g, NEWLINE_MARKER).replace(/\t/g, TAB_MARKER)
}

// Byte-level BPE keeps " cat" and "cat" as different vocabulary entries; a
// leading space is part of the token. Right-aligned labels hide it, so the
// UI shows it as an open-box glyph instead.
export const SPACE_MARKER = '␣'

export function markLeadingSpace(text: string): string {
  return text.startsWith(' ') ? SPACE_MARKER + text.slice(1) : text
}

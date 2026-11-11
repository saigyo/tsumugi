// The attention matrix fits its card: 20 px cells while they fit, smaller
// down to 8 px, after that the matrix scrolls sideways as before.
export const MAX_CELL = 20
export const MIN_CELL = 8
// below this, a label on every row and column would overlap its neighbours
export const LABEL_MIN = 14

export function cellSize(width: number | null, columns: number, chrome: number): number {
  if (width === null || columns === 0) return MAX_CELL
  return Math.max(MIN_CELL, Math.min(MAX_CELL, Math.floor((width - chrome) / columns)))
}

export const labelStep = (cell: number): number => (cell >= LABEL_MIN ? 1 : Math.ceil(LABEL_MIN / cell))

// Label margins. The labels are SVG text in the --t-tick size, so their
// width is estimated from the characters instead of measured (about 0.62 em
// each, more for capitals and the widest letters, a full em for CJK and
// emoji): the margin has room for the longest label, including special
// tokens and the folded system prompt.
// --t-tick is 0.66rem: read the root font size when measuring, so a larger browser default enlarges the margins too.
const TICK_REM = 0.66
const tickPx = (): number => {
  const root = typeof document === 'undefined' ? NaN : parseFloat(getComputedStyle(document.documentElement).fontSize)
  return TICK_REM * (root > 0 ? root : 16)
}
const GAP = 8       // between a row label and the matrix
const COL_RISE = 5  // between a column label's start and the matrix
const SLANT = Math.SQRT1_2
const glyphEm = (ch: string): number =>
  ch.charCodeAt(0) >= 0x2e80 ? 1 : /[WMmw@%]/.test(ch) ? 0.95 : /[A-Z&#]/.test(ch) ? 0.76 : 0.62
export const textWidth = (text: string): number => {
  const tick = tickPx()
  return Math.ceil([...text].reduce((w, ch) => w + glyphEm(ch) * tick, 0))
}

/** The longest label a matrix draws; a longer token is cut with an ellipsis so one long token cannot squeeze the cells. */
export const MAX_LABEL_CHARS = 22
export const shortLabel = (text: string): string => {
  const chars = [...text]
  return chars.length > MAX_LABEL_CHARS ? chars.slice(0, MAX_LABEL_CHARS - 1).join('') + '…' : text
}

export interface LabelMargins { left: number; top: number; right: number }

/**
 * The room a matrix leaves for its row labels (left), its labels rotated by
 * 45 degrees (top) and the last of those overhanging on the right.
 * `folded` is true when a column's label starts half a row higher (the folded system prompt).
 * The old fixed margins are the minimum, so short labels draw as before.
 */
export function labelMargins(rows: string[], cols: string[], opts: { folded?: boolean; minLeft: number; minTop: number; minRight: number }): LabelMargins {
  const left = Math.max(opts.minLeft, Math.max(0, ...rows.map(textWidth)) + GAP + 2)
  const rise = Math.max(0, ...cols.map((c) => textWidth(c) * SLANT + tickPx() * SLANT))
  const top = Math.max(opts.minTop, Math.ceil(rise) + COL_RISE + 4 + (opts.folded ? MAX_CELL / 2 : 0))
  // a label starting over column i reaches textWidth * SLANT to the right; MIN_CELL is the narrowest the columns can be
  const right = Math.max(opts.minRight, Math.ceil(Math.max(0, ...cols.map((c, i) => textWidth(c) * SLANT - (cols.length - i - 0.5) * MIN_CELL))) + 4)
  return { left, top, right }
}

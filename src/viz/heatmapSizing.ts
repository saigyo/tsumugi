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
// width is estimated from the characters (about 0.62 em each, a full em for
// wide characters) instead of measured: the margin always has room for the
// longest label, including special tokens and the folded system prompt.
const TICK_PX = 10.56
const GAP = 8       // between a row label and the matrix
const COL_RISE = 5  // between a column label's start and the matrix
const SLANT = Math.SQRT1_2
export const textWidth = (text: string): number =>
  Math.ceil([...text].reduce((w, ch) => w + (ch.charCodeAt(0) < 0x2e80 ? 0.62 : 1) * TICK_PX, 0))

export interface LabelMargins { left: number; top: number; right: number }

/**
 * The room a matrix leaves for its row labels (left), its labels rotated by
 * 45 degrees (top) and the last of those overhanging on the right.
 * `folded` is true when a column's label starts half a row higher (the folded system prompt).
 * The old fixed margins are the minimum, so short labels draw as before.
 */
export function labelMargins(rows: string[], cols: string[], opts: { folded?: boolean; minLeft: number; minTop: number; minRight: number }): LabelMargins {
  const left = Math.max(opts.minLeft, Math.max(0, ...rows.map(textWidth)) + GAP + 2)
  const rise = Math.max(0, ...cols.map((c) => textWidth(c) * SLANT + TICK_PX * SLANT))
  const top = Math.max(opts.minTop, Math.ceil(rise) + COL_RISE + 4 + (opts.folded ? MAX_CELL / 2 : 0))
  // a label starting over column i reaches textWidth * SLANT to the right; MIN_CELL is the narrowest the columns can be
  const right = Math.max(opts.minRight, Math.ceil(Math.max(0, ...cols.map((c, i) => textWidth(c) * SLANT - (cols.length - i - 0.5) * MIN_CELL))) + 4)
  return { left, top, right }
}

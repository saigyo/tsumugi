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

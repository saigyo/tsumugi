// Shared drawing vocabulary for vectors: the Embeddings card's row strips and
// the Logits card's readout strips use the same cells and the same colours, so
// a reader recognises "this is a row of E" across the two stages. Kept apart
// from the VectorStrip component so that file exports only components.

import { divFill } from './ramps'

// Diverging fill: indigo for positive, vermillion for negative, intensity
// relative to the strip's own peak magnitude.
export function cellFill(v: number, peak: number): string {
  return divFill(v, peak)
}

export const peakOf = (cells: number[]) => cells.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1

// A schematic row position inside the drawn E box for a token id.
export const rowMarkY = (id: number) => 40 + (id % 41)

// the pipeline band's cards never go below this share of their full side padding
const MIN_PAD_SHARE = 0.45

/** The side padding (px) that makes `cards` cards `over` pixels narrower in total, down to the smallest allowed. */
export function fittedPadding(full: number, over: number, cards: number): number {
  if (over <= 0 || cards <= 0) return full
  return Math.max(full * MIN_PAD_SHARE, Math.floor((full - over / (2 * cards)) * 10) / 10)
}

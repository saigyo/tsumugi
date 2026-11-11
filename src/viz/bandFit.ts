// the pipeline band's cards never go below this share of their full side padding
const MIN_PAD_SHARE = 0.45

/**
 * The side padding (px) that makes `cards` cards `over` pixels narrower in total, starting from the padding they
 * have `now` and never below the smallest share of the `full` padding.
 */
export function fittedPadding(now: number, over: number, cards: number, full = now): number {
  if (over <= 0 || cards <= 0) return now
  return Math.max(full * MIN_PAD_SHARE, Math.floor((now - over / (2 * cards)) * 10) / 10)
}

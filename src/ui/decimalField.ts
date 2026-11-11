// Helpers for a text field that holds a decimal number and accepts a comma or a point.

/** Comma becomes point. Returns null unless the result is empty or digits with at most one point. */
export function normaliseDecimal(text: string): string | null {
  const t = text.replace(',', '.')
  return /^\d*\.?\d*$/.test(t) ? t : null
}

/** The number in the text, brought within the limits; null when the text holds no number. */
export function decimalValue(text: string, min: number, max: number): number | null {
  const t = normaliseDecimal(text)
  if (t === null || t === '' || t === '.') return null
  const v = Number(t)
  return Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : null
}

/** Adds delta, rounds to one decimal and clamps to the limits. */
export function stepDecimal(value: number, delta: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.round((value + delta) * 10) / 10))
}

// Helpers for a text field that holds a decimal number and accepts a comma or a point.

/** Comma becomes point. Returns null unless the result is empty or digits with at most one point. */
export function normaliseDecimal(text: string): string | null {
  const t = text.replace(',', '.')
  return /^\d*\.?\d*$/.test(t) ? t : null
}

/** The number in the text, or null unless it is a finite number within the limits. */
export function decimalValue(text: string, min: number, max: number): number | null {
  const t = normaliseDecimal(text)
  if (t === null || t === '' || t === '.') return null
  const v = Number(t)
  return Number.isFinite(v) && v >= min && v <= max ? v : null
}

/** Adds delta, rounds to one decimal and clamps to the limits. */
export function stepDecimal(value: number, delta: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.round((value + delta) * 10) / 10))
}

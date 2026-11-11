// Helpers for a text field that holds a decimal number and accepts a comma or a point.

/** Trims, maps full-width digits, full stop and comma to ASCII, then turns a comma into a point. Returns null unless the result is empty or digits with at most one point. */
export function normaliseDecimal(text: string): string | null {
  const t = text.trim()
    .replace(/[\uff10-\uff19]/g, (c) => String(c.charCodeAt(0) - 0xff10))
    .replace('\uff0e', '.')
    .replace('\uff0c', ',')
    .replace(',', '.')
  return /^\d*\.?\d*$/.test(t) ? t : null
}

/** The number in the text, brought within the limits; null when the text holds no number. */
export function decimalValue(text: string, min: number, max: number): number | null {
  const t = normaliseDecimal(text)
  if (t === null || t === '' || t === '.') return null
  const v = Number(t)
  return Number.isFinite(v) ? Math.min(max, Math.max(min, v)) : null
}

/** Steps to the next multiple of a tenth in the direction of delta (as a number field with step 0.1 does), within the limits. */
export function stepDecimal(value: number, delta: number, min: number, max: number): number {
  const tenths = value * 10
  const nearest = Math.round(tenths)
  const onTenth = Math.abs(tenths - nearest) < 1e-9
  const next = delta > 0 ? (onTenth ? nearest + 1 : Math.ceil(tenths)) : (onTenth ? nearest - 1 : Math.floor(tenths))
  return Math.min(max, Math.max(min, next / 10))
}

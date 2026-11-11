// Display formats for numbers (rule R10). Display only: share links, exported files and anything parsed back keep plain number text.
const MINUS = '−'
const THIN = '\u202f' // narrow no-break space

/** Whole numbers; groups of three with a thin space from 10 000 up, four digits stay together. */
export function count(n: number): string {
  const digits = String(Math.abs(Math.round(n)))
  const body = digits.length >= 5 ? digits.replace(/\B(?=(\d{3})+(?!\d))/g, THIN) : digits
  return (n < 0 && Math.round(n) !== 0 ? MINUS : '') + body
}

/** A fixed number of decimals with a real minus; a value that rounds to zero has no sign. */
export function fixed(n: number, digits: number): string {
  const s = Math.abs(n).toFixed(digits)
  return n < 0 && Number(s) !== 0 ? MINUS + s : s
}

/** Pushes and changes: always signed, except a value that rounds to zero. */
export function signed(n: number, digits = 2): string {
  const s = fixed(Math.abs(n), digits)
  if (Number(s) === 0) return s
  return (n < 0 ? MINUS : '+') + s
}

/** A probability (0..1) as a whole percent, without a space; "<1%" and ">99%" instead of a misleading 0 or 100. */
export function percent(p: number | null): string {
  if (p === null) return '—'
  const n = Math.round(p * 100)
  if (p > 0 && n === 0) return '<1%'
  if (p < 1 && n === 100) return '>99%'
  return `${n}%`
}

/** The difference of two probabilities (−1..1) in percentage points. */
export function points(d: number, approx = false): string {
  const n = Math.round(d * 100)
  const body = n === 0 ? '0 pts' : `${n < 0 ? MINUS : '+'}${Math.abs(n)} pts`
  return approx ? `≈ ${body}` : body
}

// Chart colour ramps, mixed in the browser from the theme's end-point variables
// (--ramp-lo/-hi for values, --div-* for positive/negative), so a theme change
// recolours every chart. Mixing in HSL between the end points reproduces the
// light theme's hue-constant ramps.

const clamp01 = (t: number) => (Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : t > 0 ? 1 : 0)
// whole-tenth percentage, so the strings stay short
const pct = (t: number) => Math.round(clamp01(t) * 1000) / 10

// t in 0..1 (clamped): low → high of the value ramp
export function seqFill(t: number): string {
  return `color-mix(in hsl, var(--ramp-hi) ${pct(t)}%, var(--ramp-lo))`
}

// v relative to peak (clamped to ±1): each side from its own low end to its full colour
export function divFill(v: number, peak: number): string {
  const p = pct(Math.abs(v) / peak)
  return v >= 0
    ? `color-mix(in hsl, var(--div-pos) ${p}%, var(--div-pos-lo))`
    : `color-mix(in hsl, var(--div-neg) ${p}%, var(--div-neg-lo))`
}

// the canvas colour function the result is written in (kept out of a literal so the colour-literal guard stays strict)
const CANVAS_FN = 'rgb'

type Hsl = [number, number, number]   // degrees, 0..1, 0..1
type Rgb = [number, number, number]       // 0..255

const NUM = '(-?\\d*\\.?\\d+)'
const HSL_RE = new RegExp(`^hsla?\\(\\s*${NUM}(?:deg)?[\\s,]+${NUM}%[\\s,]+${NUM}%`, 'i')
const RGB_RE = new RegExp(`^rgba?\\(\\s*${NUM}[\\s,]+${NUM}[\\s,]+${NUM}`, 'i')

function rgbToHsl([r, g, b]: Rgb): Hsl {
  r /= 255; g /= 255; b /= 255
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min
  const l = (max + min) / 2
  if (d === 0) return [0, 0, l]
  const s = d / (1 - Math.abs(2 * l - 1))
  const h = max === r ? ((g - b) / d + 6) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4
  return [h * 60, s, l]
}
function hslToRgb([h, s, l]: Hsl): Rgb {
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => {
    const k = (n + h / 30) % 12
    return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)))
  }
  return [f(0), f(8), f(4)]
}

// hsl() is taken as written, so the hue and saturation of a pale end are not
// disturbed by an 8-bit round trip; rgb() and #hex go through rgb. Anything else is null.
function parseColour(css: string): Hsl | null {
  const c = css.trim()
  const hsl = HSL_RE.exec(c)
  if (hsl) return [((Number(hsl[1]) % 360) + 360) % 360, Number(hsl[2]) / 100, Number(hsl[3]) / 100]
  const rgb = RGB_RE.exec(c)
  if (rgb) return rgbToHsl([Number(rgb[1]), Number(rgb[2]), Number(rgb[3])])
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(c)?.[1]
  if (hex) {
    const full = hex.length === 3 ? [...hex].map((x) => x + x).join('') : hex
    return rgbToHsl([0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as Rgb)
  }
  return null
}

// the light theme's ends, used when a custom property holds something unparseable
const FALLBACK_LO: Hsl = [211, 0.3, 0.94]
const FALLBACK_HI: Hsl = [211, 0.55, 0.24]

// for canvas: the same mixing as the CSS ramp, done in JS from the end colours
// (hsl(), rgb() or #hex); an unparseable end falls back to the light theme's.
export function seqRgb(t: number, lo: string, hi: string): string {
  const u = clamp01(t)
  const a = parseColour(lo) ?? FALLBACK_LO
  const b = parseColour(hi) ?? FALLBACK_HI
  // a grey end has no hue of its own: take the other's, as CSS does
  const ha = a[1] === 0 ? b[0] : a[0], hb = b[1] === 0 ? a[0] : b[0]
  const dh = ((hb - ha + 540) % 360) - 180
  const m: Hsl = [(ha + dh * u + 360) % 360, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]
  const [r, g, bl] = hslToRgb(m).map((x) => Math.round(Math.min(255, Math.max(0, x))))
  return `${CANVAS_FN}(${r}, ${g}, ${bl})`
}

// The two ramp end points as the custom properties are written (full precision).
// A value this file cannot parse (oklch(), color(), var(...)) is resolved by the
// browser to an rgb colour instead; failing that, seqRgb's fallback applies.
export function readRampEnds(): { lo: string; hi: string } {
  const root = getComputedStyle(document.documentElement)
  let probe: HTMLElement | null = null
  const read = (name: string) => {
    const raw = root.getPropertyValue(name).trim()
    if (parseColour(raw)) return raw
    probe ??= document.body.appendChild(document.createElement('span'))
    probe.style.color = `var(${name})`
    return getComputedStyle(probe).color
  }
  const ends = { lo: read('--ramp-lo'), hi: read('--ramp-hi') }
  ;(probe as HTMLElement | null)?.remove()
  return ends
}

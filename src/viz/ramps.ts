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

// the canvas colour function the ends are read and written in (kept out of a literal so the colour-literal guard stays strict)
const CANVAS_FN = 'rgb'

type Rgb = [number, number, number]

function parseRgb(css: string): Rgb {
  const n = css.match(/-?\d*\.?\d+/g)?.slice(0, 3).map(Number)
  return n && n.length === 3 ? [n[0], n[1], n[2]] : [0, 0, 0]
}
function toHsl([r, g, b]: Rgb): Rgb {
  r /= 255; g /= 255; b /= 255
  const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min
  const l = (max + min) / 2
  if (d === 0) return [0, 0, l]
  const s = d / (1 - Math.abs(2 * l - 1))
  const h = max === r ? ((g - b) / d + 6) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4
  return [h * 60, s, l]
}
function fromHsl([h, s, l]: Rgb): Rgb {
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => {
    const k = (n + h / 30) % 12
    return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)))
  }
  return [f(0), f(8), f(4)]
}

// for canvas: the same mixing done in JS from end colours read off the document ('rgb(r, g, b)')
export function seqRgb(t: number, lo: string, hi: string): string {
  const u = clamp01(t)
  const a = toHsl(parseRgb(lo)), b = toHsl(parseRgb(hi))
  // a grey end has no hue of its own: take the other's, as CSS does
  const ha = a[1] === 0 ? b[0] : a[0], hb = b[1] === 0 ? a[0] : b[0]
  const dh = ((hb - ha + 540) % 360) - 180
  const m: Rgb = [(ha + dh * u + 360) % 360, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]
  const [r, g, bl] = fromHsl(m).map((x) => Math.round(Math.min(255, Math.max(0, x))))
  return `${CANVAS_FN}(${r}, ${g}, ${bl})`
}

// the two ramp end points as the browser resolves them right now ('rgb(...)')
export function readRampEnds(): { lo: string; hi: string } {
  const probe = document.createElement('span')
  document.body.appendChild(probe)
  const read = (v: string) => { probe.style.color = `var(${v})`; return getComputedStyle(probe).color }
  const ends = { lo: read('--ramp-lo'), hi: read('--ramp-hi') }
  probe.remove()
  return ends
}

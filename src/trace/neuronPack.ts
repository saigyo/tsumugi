// Neuron data rides in the trace as base64: activations as float16 (half the
// bytes of float32, about 3 significant digits — the card shows one decimal)
// and candidate rows as float32 (the exact readout rows). Little-endian, like
// every engine we run on.

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)

function toHalf(v: number): number {
  if (Number.isFinite(v) && Math.abs(v) > 65504) return (v < 0 ? 0x8000 : 0) | 0x7bff  // clamp to ±65504, the largest float16
  f32[0] = v
  const x = u32[0]
  const sign = (x >>> 16) & 0x8000
  const exp = (x >>> 23) & 0xff
  let mant = x & 0x7fffff
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0)          // infinity, NaN
  const e = exp - 127 + 15
  if (e >= 0x1f) return sign | 0x7c00                                     // too large: infinity
  if (e <= 0) {                                                           // subnormal half, or zero
    if (e < -10) return sign
    mant |= 0x800000
    const shift = 14 - e
    let half = mant >> shift
    const rem = mant & ((1 << shift) - 1), mid = 1 << (shift - 1)
    if (rem > mid || (rem === mid && (half & 1))) half++
    return sign | half
  }
  let half = sign | (e << 10) | (mant >> 13)
  const rem = mant & 0x1fff
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1))) half++             // round to nearest even; a carry
  return half                                                             // into the exponent is correct
}

function fromHalf(h: number): number {
  const sign = h & 0x8000 ? -1 : 1
  const exp = (h >> 10) & 0x1f
  const mant = h & 0x3ff
  if (exp === 0) return sign * mant * 2 ** -24
  if (exp === 0x1f) return mant ? NaN : sign * Infinity
  return sign * (1 + mant / 1024) * 2 ** (exp - 15)
}

function toBase64(bytes: Uint8Array): string {
  let s = ''
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(s)
}

function fromBase64(b64: string): Uint8Array {
  const s = atob(b64)
  const out = new Uint8Array(s.length)
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i)
  return out
}

export function packHalf(values: ArrayLike<number>): string {
  const u = new Uint16Array(values.length)
  for (let i = 0; i < values.length; i++) u[i] = toHalf(values[i])
  return toBase64(new Uint8Array(u.buffer))
}

export function unpackHalf(b64: string): Float32Array {
  const bytes = fromBase64(b64)
  if (bytes.length % 2) throw new Error('odd byte count for float16 data')
  const u = new Uint16Array(bytes.buffer, 0, bytes.length / 2)
  const out = new Float32Array(u.length)
  for (let i = 0; i < u.length; i++) out[i] = fromHalf(u[i])
  return out
}

export function packFloat32(values: ArrayLike<number>): string {
  return toBase64(new Uint8Array(Float32Array.from(values).buffer))
}

export function unpackFloat32(b64: string): Float32Array {
  const bytes = fromBase64(b64)
  if (bytes.length % 4) throw new Error('float32 data is not a multiple of 4 bytes')
  return new Float32Array(bytes.buffer, 0, bytes.length / 4)
}

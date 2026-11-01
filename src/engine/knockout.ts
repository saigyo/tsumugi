/** Per-layer MLP multiplier: all ones, or zero at layer `off` (the knocked-out MLP). */
export function mlpScale(layers: number, off: number | null): Float32Array {
  const s = new Float32Array(layers).fill(1)
  if (off !== null && off >= 0 && off < layers) s[off] = 0
  return s
}

/** The copied-prefix token to force at `cycle`, or null once the prefix is used up. */
export function forcedToken(forced: number[], cycle: number): number | null {
  return cycle >= 0 && cycle < forced.length ? forced[cycle] : null
}

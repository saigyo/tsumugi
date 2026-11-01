import type { GenParams } from '../trace/types'
import { sampleIndex, softmax, topK } from './math'

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

/**
 * One sweep pass's reading of the last-position logits: the chosen token's share among the top-k
 * (0 if it fell out), the new top token, and whether the run's own sampler would now pick
 * something else (greedy: top id differs; otherwise the run's fixed draw lands elsewhere).
 */
export function sweepEntry(lastLogits: Float32Array, params: GenParams, chosenId: number, draw: number):
  { prob: number; topId: number; topProb: number; flips: boolean } {
  const top = topK(lastLogits, params.topK)
  const probs = softmax(top.map((c) => c.logit), params.temperature || 1)
  const at = top.findIndex((c) => c.id === chosenId)
  const picked = params.temperature === 0 ? top[0].id : top[sampleIndex(probs, () => draw)].id
  return { prob: at < 0 ? 0 : probs[at], topId: top[0].id, topProb: probs[0], flips: picked !== chosenId }
}

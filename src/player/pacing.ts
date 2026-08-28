import type { TraceEvent } from '../trace/types'

export const BASE_MS = 600

const MULTIPLIER: Record<TraceEvent['type'], number> = {
  'run-start': 0.5, tokenize: 1.5, embed: 1.5, layer: 0.2, attention: 2,
  logits: 1.5, softmax: 1.5, sample: 2.5, append: 1.5, 'attention-grid': 0.5, 'run-end': 0.5,
}

export function delayFor(e: TraceEvent | undefined, speed: number): number {
  const mult = e ? MULTIPLIER[e.type] : 1
  return (BASE_MS * mult) / speed
}

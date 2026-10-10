import type { TraceEvent } from '../trace/types'
import { latestOfType } from '../viz/selectors'
import type { RunSummary } from './types'

// Plain values for "In this run" blocks: run-level facts from the whole trace,
// per-cycle facts at the cursor. Null when no run is loaded.
export function runSummary(events: TraceEvent[], cursor: number): RunSummary | null {
  const last = events.length - 1
  const start = latestOfType(events, last, 'run-start')
  if (!start) return null
  const at = Math.max(0, Math.min(cursor, last))
  let cycle = 0
  let maxCycle = -1
  events.forEach((e, i) => {
    if (!('cycle' in e)) return
    if (i <= at) cycle = e.cycle
    maxCycle = Math.max(maxCycle, e.cycle)
  })
  const tokenize = latestOfType(events, last, 'tokenize')
  const embed = latestOfType(events, at, 'embed')
  const layer = latestOfType(events, last, 'layer')
  const sm = latestOfType(events, at, 'softmax')
  const lg = latestOfType(events, at, 'logits')
  const logitOf = new Map((lg && sm && lg.cycle === sm.cycle ? lg.topK : []).map((c) => [c.id, c.logit]))
  return {
    mode: start.mode,
    modelId: start.modelId,
    params: start.params,
    tokens: tokenize?.tokens ?? [],
    cycle,
    cycles: maxCycle + 1,
    ...(embed ? { stream: { seqLen: embed.seqLen, dims: embed.dims, layers: layer?.total ?? 0 } } : {}),
    ...(start.vocabSize ? { vocabSize: start.vocabSize } : {}),
    ...(sm ? { softmax: { temperature: sm.temperature,
      top: sm.topK.map((c) => ({ text: c.text, logit: logitOf.get(c.id), prob: c.prob })) } } : {}),
  }
}

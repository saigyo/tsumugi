import type { KnockoutSweep, TraceEvent } from '../trace/types'
import { cycleVotes, editRows, latestOfType } from '../viz/selectors'
import type { RunSummary } from './types'

// Plain values for "In this run" blocks: run-level facts from the whole trace,
// per-cycle facts at the cursor. Null when no run is loaded. `sweeps` are the saved
// knockout sweeps of the record these events belong to; they sit beside the events,
// not in them, so the caller passes them in.
export function runSummary(events: TraceEvent[], cursor: number, sweeps?: KnockoutSweep[]): RunSummary | null {
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
  const end = latestOfType(events, last, 'run-end')
  const tokenize = latestOfType(events, last, 'tokenize')
  const embed = latestOfType(events, at, 'embed')
  const layer = latestOfType(events, last, 'layer')
  const sm = latestOfType(events, at, 'softmax')
  const lg = latestOfType(events, at, 'logits')
  const logitOf = new Map((lg && sm && lg.cycle === sm.cycle ? lg.topK : []).map((c) => [c.id, c.logit]))
  const ed = editRows(events, at)
  const lastRow = ed?.mlp.at(-1)
  const edits = ed && lastRow
    ? (() => { const layer = lastRow.indexOf(Math.max(...lastRow))
        return { token: ed.tokens.at(-1)!, layer, ratio: lastRow[layer] } })()
    : undefined
  const cv = cycleVotes(events, at)
  const votes = cv.status === 'ok'
    ? (() => { const top = cv.candidates[0]; const layer = top.votes.mlp.indexOf(Math.max(...top.votes.mlp))
        return { candidate: { id: top.id, text: top.text }, layer, value: top.votes.mlp[layer] } })()
    : undefined
  const sw = sweeps?.find((s) => s.cycle === cycle)
  const knockout = sw ? knockoutSummary(sw) : undefined
  return {
    mode: start.mode,
    modelId: start.modelId,
    params: start.params,
    tokens: tokenize?.tokens ?? [],
    cycle,
    cycles: maxCycle + 1,
    generated: events.filter((e) => e.type === 'append').length,
    ...(end ? { ended: end.reason } : {}),
    ...(embed ? { stream: { seqLen: embed.seqLen, dims: embed.dims, layers: layer?.total ?? 0 } } : {}),
    ...(start.vocabSize ? { vocabSize: start.vocabSize } : {}),
    ...(sm ? { softmax: { temperature: sm.temperature,
      top: sm.topK.map((c) => ({ id: c.id, text: c.text, logit: logitOf.get(c.id), prob: c.prob })) } } : {}),
    ...(edits ? { edits } : {}),
    ...(votes ? { votes } : {}),
    ...(knockout ? { knockout } : {}),
  }
}

function knockoutSummary(sw: KnockoutSweep): NonNullable<RunSummary['knockout']> {
  let drop: { layer: number; delta: number } | undefined
  let rise: { layer: number; delta: number } | undefined
  for (const l of sw.layers) {
    const d = sw.baseline - l.prob
    if (d > 0 && (!drop || d > drop.delta)) drop = { layer: l.layer, delta: d }
    if (d < 0 && (!rise || -d > rise.delta)) rise = { layer: l.layer, delta: -d }
  }
  return { chosen: sw.chosen, baseline: sw.baseline, layers: sw.layers.length,
    flipped: sw.layers.filter((l) => l.flips).length, ...(drop ? { drop } : {}), ...(rise ? { rise } : {}) }
}

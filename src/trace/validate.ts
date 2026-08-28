import type { TraceEvent } from './types'

export function validateTrace(events: TraceEvent[]): string[] {
  const errs: string[] = []
  if (events[0]?.type !== 'run-start') errs.push('first event must be run-start')
  if (events[1]?.type !== 'tokenize') errs.push('second event must be tokenize')
  if (events[events.length - 1]?.type !== 'run-end') errs.push('last event must be run-end')

  const CYCLE = ['embed', 'layer', 'logits', 'softmax', 'sample', 'append'] as const
  let phase: (typeof CYCLE)[number] = 'embed'
  let layerIdx = 0
  for (const e of events.slice(2, -1)) {
    if (e.type === 'attention-grid') continue  // placement checked below
    if (e.type === 'layer') {
      if (phase !== 'layer' && phase !== 'logits') { errs.push(`unexpected layer in phase ${phase}`); continue }
      if (e.index !== layerIdx) errs.push(`layer index ${e.index}, expected ${layerIdx}`)
      layerIdx++
      phase = layerIdx >= e.total ? 'logits' : 'layer'
      continue
    }
    if (e.type === 'attention') {
      // optional slot: after the cycle's layers, before its logits
      if (phase !== 'logits') { errs.push(`unexpected attention in phase ${phase}`); continue }
      for (const h of e.heads) {
        h.matrix.forEach((row, i) => {
          if (row.length !== i + 1) errs.push(`attention row ${i} not causal (length ${row.length})`)
          const sum = row.reduce((a, b) => a + b, 0)
          if (Math.abs(sum - 1) > 1e-4) errs.push(`attention row ${i} sums to ${sum}`)
        })
      }
      continue
    }
    if (e.type === 'embed' && phase === 'embed') { phase = 'layer'; layerIdx = 0; continue }
    if (e.type === phase) {
      phase = CYCLE[(CYCLE.indexOf(phase) + 1) % CYCLE.length]
      continue
    }
    errs.push(`unexpected ${e.type} in phase ${phase}`)
  }

  const grids = events.flatMap((e, i) => (e.type === 'attention-grid' ? [{ e, i }] : []))
  if (grids.length > 1) errs.push('more than one attention-grid event')
  for (const { e, i } of grids) {
    if (i !== events.length - 2) errs.push('attention-grid must sit directly before run-end')
    if (e.cells.length !== e.layers * e.heads)
      errs.push(`attention-grid has ${e.cells.length} cells, expected layers × heads = ${e.layers * e.heads}`)
    for (const c of e.cells) {
      if (c.thumb.length > 12 || c.thumb.some((r) => r.length > 12))
        errs.push(`attention-grid thumb L${c.layer}·H${c.head} exceeds 12×12`)
      if (c.thumb.some((r) => r.some((v) => v < 0 || v > 1)))
        errs.push(`attention-grid thumb L${c.layer}·H${c.head} has values outside [0, 1]`)
    }
  }

  for (const e of events) {
    if (e.type === 'softmax') {
      const sum = e.topK.reduce((a, c) => a + c.prob, 0)
      if (Math.abs(sum - 1) > 1e-4) errs.push(`softmax probs sum to ${sum}`)
      for (let i = 1; i < e.topK.length; i++)
        if (e.topK[i].prob > e.topK[i - 1].prob) errs.push('softmax topK not sorted desc')
    }
    if (e.type === 'logits')
      for (let i = 1; i < e.topK.length; i++)
        if (e.topK[i].logit > e.topK[i - 1].logit) errs.push('logits topK not sorted desc')
  }
  return errs
}

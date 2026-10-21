import { unpackFloat32, unpackHalf } from './neuronPack'
import { READOUT_CELLS, type MlpSteps, type NeuronData, type TraceEvent } from './types'

export function validateTrace(events: TraceEvent[]): string[] {
  const errs: string[] = []
  if (events[0]?.type !== 'run-start') errs.push('first event must be run-start')
  if (events[1]?.type !== 'tokenize') errs.push('second event must be tokenize')
  if (events[events.length - 1]?.type !== 'run-end') errs.push('last event must be run-end')

  const CYCLE = ['embed', 'layer', 'logits', 'softmax', 'sample', 'append'] as const
  let phase: (typeof CYCLE)[number] = 'embed'
  let layerIdx = 0
  let layerTotal = 0
  let seqLen = 0
  let dims = 0
  for (const e of events.slice(2, -1)) {
    if (e.type === 'attention-grid') continue  // placement checked below
    if (e.type === 'layer') {
      if (phase !== 'layer' && phase !== 'logits') { errs.push(`unexpected layer in phase ${phase}`); continue }
      if (e.index !== layerIdx) errs.push(`layer index ${e.index}, expected ${layerIdx}`)
      layerTotal = e.total
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
    if (e.type === 'edits') {
      // optional slot, like attention: after the cycle's layers, before its logits
      if (phase !== 'logits') { errs.push(`unexpected edits in phase ${phase}`); continue }
      const ok = (r: unknown) => Array.isArray(r) && r.length === layerTotal
        && r.every((v) => typeof v === 'number' && Number.isFinite(v))
      if (e.attn.length !== e.mlp.length)
        errs.push(`edits cycle ${e.cycle} has ${e.attn.length} attn rows but ${e.mlp.length} mlp rows`)
      e.attn.forEach((r, i) => { if (!ok(r)) errs.push(`edits cycle ${e.cycle} attn row ${i} is not ${layerTotal} finite numbers`) })
      e.mlp.forEach((r, i) => { if (!ok(r)) errs.push(`edits cycle ${e.cycle} mlp row ${i} is not ${layerTotal} finite numbers`) })
      if (e.start < 0 || e.start + e.attn.length > seqLen)
        errs.push(`edits cycle ${e.cycle} rows ${e.start}..${e.start + e.attn.length} exceed seqLen ${seqLen}`)
      if (e.neurons !== undefined)
        for (const p of neuronProblems(e.neurons, layerTotal, dims)) errs.push(`edits cycle ${e.cycle} ${p}`)
      if (e.neurons?.steps !== undefined)
        for (const p of stepProblems(e.neurons.steps, layerTotal, e.neurons.neurons, dims)) errs.push(`edits cycle ${e.cycle} ${p}`)
      continue
    }
    if (e.type === 'embed' && phase === 'embed') {
      seqLen = e.seqLen
      dims = e.dims
      if (e.rows) {
        e.rows.forEach((row, i) => {
          if (!Array.isArray(row) || row.length !== e.dims || row.some((v) => typeof v !== 'number' || !Number.isFinite(v)))
            errs.push(`embed cycle ${e.cycle} row ${i} is not ${e.dims} finite numbers`)
        })
      }
      if (e.source === 'model' && !e.rows) errs.push(`embed cycle ${e.cycle} claims model source without rows`)
      phase = 'layer'; layerIdx = 0; continue
    }
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
    if (e.type === 'logits') {
      for (let i = 1; i < e.topK.length; i++)
        if (e.topK[i].logit > e.topK[i - 1].logit) errs.push('logits topK not sorted desc')
      const finite = (a: unknown): a is number[] =>
        Array.isArray(a) && a.every((v) => typeof v === 'number' && Number.isFinite(v))
      if (e.lastHidden !== undefined && !finite(e.lastHidden))
        errs.push(`logits cycle ${e.cycle} lastHidden is not an array of finite numbers`)
      for (const c of e.topK) {
        for (const key of ['row', 'dot'] as const) {
          const v = c[key]
          if (v !== undefined && (!finite(v) || v.length !== READOUT_CELLS))
            errs.push(`logits cycle ${e.cycle} candidate ${c.id} ${key} is not ${READOUT_CELLS} finite numbers`)
        }
        const v = c.votes
        if (v !== undefined && (typeof v.embed !== 'number' || !Number.isFinite(v.embed)
          || !finite(v.attn) || !finite(v.mlp) || v.attn.length !== layerTotal || v.mlp.length !== layerTotal))
          errs.push(`logits cycle ${e.cycle} candidate ${c.id} votes are not ${layerTotal} finite numbers per block`)
      }
    }
  }
  return errs
}

export function neuronProblems(n: unknown, layers: number, dims: number): string[] {
  if (typeof n !== 'object' || n === null) return ['neurons is not an object']
  const d = n as Partial<NeuronData>
  const out: string[] = []
  const count = d.neurons
  if (typeof count !== 'number' || !Number.isInteger(count) || count <= 0) out.push('neurons count is not a positive integer')
  if (typeof d.invRms !== 'number' || !Number.isFinite(d.invRms) || d.invRms <= 0) out.push('invRms is not a positive number')
  try {
    const a = unpackHalf(String(d.activations))
    if (a.length !== layers * (count ?? 0)) out.push(`activations hold ${a.length} values, expected ${layers} × ${count}`)
    else if (a.some((v) => !Number.isFinite(v))) out.push('activations are not all finite')
  } catch { out.push('activations are not base64 float16') }
  if (!Array.isArray(d.candidates) || d.candidates.length === 0) out.push('candidates are missing')
  else for (const c of d.candidates) {
    const id = (c as { id?: unknown })?.id
    try {
      const row = typeof id === 'number' ? unpackFloat32(String(c.row)) : null
      if (!row || row.length !== dims) out.push(`candidate ${String(id)} row is not ${dims} float32 values`)
      else if (row.some((v) => !Number.isFinite(v))) out.push(`candidate ${id} row is not all finite`)
    } catch { out.push(`candidate ${String(id)} row is not base64 float32`) }
  }
  return out
}

export function stepProblems(s: unknown, layers: number, neurons: number, dims: number): string[] {
  if (typeof s !== 'object' || s === null) return ['steps is not an object']
  const d = s as Partial<Record<keyof MlpSteps, unknown>>
  const out: string[] = []
  const field = (k: keyof MlpSteps, width: number, label: string) => {
    try {
      const v = unpackHalf(String(d[k]))
      if (v.length !== layers * width) out.push(`steps.${k} holds ${v.length} values, expected ${layers} × ${label}`)
      else if (v.some((x) => !Number.isFinite(x))) out.push(`steps.${k} values are not all finite`)
    } catch { out.push(`steps.${k} is not base64 float16`) }
  }
  field('gate', neurons, String(neurons)); field('up', neurons, String(neurons))
  field('mlpIn', dims, String(dims)); field('mlpOut', dims, String(dims))
  return out
}

// An import keeps its run when only neuron data is damaged: the broken fields
// are dropped (neuron data whole, or only its steps) and counted per kind.
export function dropBrokenNeurons(events: TraceEvent[]): { events: TraceEvent[]; droppedNeurons: number; droppedSteps: number } {
  let layers = 0, dims = 0, droppedNeurons = 0, droppedSteps = 0
  const out = events.map((e) => {
    if (e.type === 'layer') layers = e.total
    if (e.type === 'embed') dims = e.dims
    if (e.type !== 'edits' || e.neurons === undefined) return e
    if (neuronProblems(e.neurons, layers, dims).length > 0) {
      droppedNeurons++
      const rest = { ...e }
      delete rest.neurons
      return rest
    }
    if (e.neurons.steps === undefined || stepProblems(e.neurons.steps, layers, e.neurons.neurons, dims).length === 0) return e
    droppedSteps++
    const neurons = { ...e.neurons }
    delete neurons.steps
    return { ...e, neurons }
  })
  return { events: out, droppedNeurons, droppedSteps }
}

const isInt = (x: unknown): x is number => Number.isInteger(x)
const isPair = (x: unknown): x is [number, number] => Array.isArray(x) && x.length === 2 && isInt(x[0]) && isInt(x[1])

// A tokenize event's chat spans are trusted only when they fit its tokens:
// user 0 <= u0 <= u1 <= tokens, and a system turn 1 <= s0 < s1 <= u0.
function chatSpansOk(chat: unknown, tokens: number): boolean {
  if (typeof chat !== 'object' || chat === null) return false
  const { user, system } = chat as { user?: unknown; system?: unknown }
  if (!isPair(user) || user[0] < 0 || user[0] > user[1] || user[1] > tokens) return false
  return system === undefined || (isPair(system) && system[0] >= 1 && system[0] < system[1] && system[1] <= user[0])
}

// An import keeps its run when only the chat spans are damaged: the spans are
// dropped (the run shows unfolded) and counted.
export function dropBrokenChat(events: TraceEvent[]): { events: TraceEvent[]; droppedChats: number } {
  let droppedChats = 0
  const out = events.map((e) => {
    if (e.type !== 'tokenize' || !('chat' in e)) return e
    if (Array.isArray(e.tokens) && e.chat !== undefined && chatSpansOk(e.chat, e.tokens.length)) return e
    droppedChats++
    const rest = { ...e }
    delete rest.chat
    return rest
  })
  return { events: out, droppedChats }
}

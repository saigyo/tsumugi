import { expect, test } from 'vitest'
import { buildFixtureTrace, fixtureReadout, makeFixtureTrace, makeGridEvent } from '../test/fixtures'
import { dropBrokenNeurons, neuronProblems, stepProblems, validateTrace } from './validate'
import { packFloat32, packHalf } from './neuronPack'
import { READOUT_CELLS, type MlpSteps, type NeuronData, type TraceEvent } from './types'

test('fixture trace is valid', () => {
  expect(validateTrace(makeFixtureTrace())).toEqual([])
})

test('missing run-start is flagged', () => {
  expect(validateTrace(makeFixtureTrace().slice(1)).length).toBeGreaterThan(0)
})

test('out-of-order layer indices are flagged', () => {
  const t = makeFixtureTrace()
  const i = t.findIndex((e) => e.type === 'layer')
  const j = t.findIndex((e, k) => e.type === 'layer' && k > i)
  ;[t[i], t[j]] = [t[j], t[i]]
  expect(validateTrace(t).some((v) => v.includes('layer'))).toBe(true)
})

test('softmax probs must sum to 1', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'softmax')
  if (e?.type === 'softmax') e.topK = e.topK.map((c) => ({ ...c, prob: c.prob * 2 }))
  expect(validateTrace(t).some((v) => v.includes('softmax'))).toBe(true)
})

test('attention rows must be causal and sum to 1', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'attention')
  if (e?.type === 'attention') e.heads[0].matrix[1] = [0.9, 0.9]
  expect(validateTrace(t).some((v) => v.includes('attention'))).toBe(true)
})

test('attention outside the post-layer slot is flagged', () => {
  const t = makeFixtureTrace()
  const i = t.findIndex((e) => e.type === 'attention')
  const [att] = t.splice(i, 1)
  t.splice(2, 0, att)  // before embed
  expect(validateTrace(t).some((v) => v.includes('attention'))).toBe(true)
})

test('default fixture has no attention-grid event', () => {
  expect(makeFixtureTrace().some((e) => e.type === 'attention-grid')).toBe(false)
})

test('attention-grid directly before run-end is valid', () => {
  const t = makeFixtureTrace()
  t.splice(t.length - 1, 0, makeGridEvent())
  expect(validateTrace(t)).toEqual([])
})

test('attention-grid away from run-end is flagged', () => {
  const t = makeFixtureTrace()
  t.splice(2, 0, makeGridEvent())
  expect(validateTrace(t).some((v) => v.includes('attention-grid'))).toBe(true)
})

test('attention-grid cell count must be layers × heads', () => {
  const t = makeFixtureTrace()
  const g = makeGridEvent(2, 2)
  g.cells.pop()
  t.splice(t.length - 1, 0, g)
  expect(validateTrace(t).some((v) => v.includes('cells'))).toBe(true)
})

test('attention-grid thumbs must be ≤12×12 with values in [0,1]', () => {
  const t = makeFixtureTrace()
  const g = makeGridEvent(2, 2)
  g.cells[0].thumb[0][0] = 1.5
  g.cells[1].thumb = Array.from({ length: 13 }, () => Array.from({ length: 13 }, () => 0))
  t.splice(t.length - 1, 0, g)
  const errs = validateTrace(t)
  expect(errs.some((v) => v.includes('[0, 1]'))).toBe(true)
  expect(errs.some((v) => v.includes('12'))).toBe(true)
})

test('embed rows must be dims-long finite numbers', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'embed')
  if (e?.type === 'embed') { e.source = 'model'; e.rows = [[0.1, 0.2]] }   // dims is 576
  expect(validateTrace(t).some((v) => v.includes('embed'))).toBe(true)
})

test('model-source embed without rows is flagged', () => {
  const t = makeFixtureTrace()
  const e = t.find((x) => x.type === 'embed')
  if (e?.type === 'embed') e.source = 'model'
  expect(validateTrace(t).some((v) => v.includes('embed'))).toBe(true)
})

test('model-source embed with well-formed rows is valid', () => {
  expect(validateTrace(buildFixtureTrace({ embedRows: true }))).toEqual([])
})

test('legacy embed events carrying preview and no source still validate', () => {
  const t = makeFixtureTrace()
  const i = t.findIndex((x) => x.type === 'embed')
  t[i] = { type: 'embed', cycle: 0, seqLen: 2, dims: 576, preview: [[0.1, 0.2]] } as unknown as TraceEvent
  expect(validateTrace(t)).toEqual([])
})

test('readout fields on logits events validate, and dot cells sum to the logit', () => {
  const t = buildFixtureTrace({ readout: true })
  expect(validateTrace(t)).toEqual([])
  const logits = t.find((e) => e.type === 'logits')
  if (logits?.type !== 'logits') throw new Error('no logits event')
  expect(logits.lastHidden).toHaveLength(576)
  for (const c of logits.topK) {
    expect(c.row).toHaveLength(READOUT_CELLS)
    expect(c.dot).toHaveLength(READOUT_CELLS)
    expect(c.dot!.reduce((a, b) => a + b, 0)).toBeCloseTo(c.logit, 6)
  }
})

test('readout fields with wrong length are flagged', () => {
  const t = buildFixtureTrace({ readout: true })
  const logits = t.find((e) => e.type === 'logits')
  if (logits?.type !== 'logits') throw new Error('no logits event')
  logits.topK[0].dot = [1, 2, 3]
  logits.lastHidden = [1, Number.NaN]
  const errs = validateTrace(t)
  expect(errs.some((e) => /dot/.test(e) && /64/.test(e))).toBe(true)
  expect(errs.some((e) => /lastHidden/.test(e))).toBe(true)
})

test('logits events without readout fields stay valid', () => {
  expect(validateTrace(buildFixtureTrace())).toEqual([])
})

test('fixtureReadout is deterministic and sums to the logit', () => {
  expect(fixtureReadout(42, 9.1)).toEqual(fixtureReadout(42, 9.1))
  expect(fixtureReadout(42, 9.1).dot.reduce((a, b) => a + b, 0)).toBeCloseTo(9.1, 6)
})

test('a trace with MLP data validates; old traces still do', () => {
  expect(validateTrace(buildFixtureTrace({ mlp: true, readout: true }))).toEqual([])
  expect(validateTrace(buildFixtureTrace())).toEqual([])
})

test('edits rows must be one value per layer and fit the sequence', () => {
  const t = buildFixtureTrace({ mlp: true })
  const i = t.findIndex((e) => e.type === 'edits')
  const e = t[i] as Extract<TraceEvent, { type: 'edits' }>
  t[i] = { ...e, attn: e.attn.map((r) => r.slice(1)) }
  expect(validateTrace(t).join()).toMatch(/edits cycle 0 attn row 0 is not 3 finite numbers/)
  t[i] = { ...e, start: 5 }
  expect(validateTrace(t).join()).toMatch(/edits cycle 0 rows 5\.\.7 exceed seqLen 2/)
  t[i] = { ...e, mlp: e.mlp.slice(1) }
  expect(validateTrace(t).join()).toMatch(/edits cycle 0 has 2 attn rows but 1 mlp rows/)
})

test('edits sit after the layers, before logits', () => {
  const t = buildFixtureTrace({ mlp: true })
  const i = t.findIndex((e) => e.type === 'edits')
  const [ev] = t.splice(i, 1)
  t.splice(3, 0, ev)   // right after embed, before the layers
  expect(validateTrace(t).join()).toMatch(/unexpected edits in phase layer/)
})

test('votes need one value per layer', () => {
  const t = buildFixtureTrace({ mlp: true })
  const lg = t.find((e) => e.type === 'logits') as Extract<TraceEvent, { type: 'logits' }>
  lg.topK[0] = { ...lg.topK[0], votes: { ...lg.topK[0].votes!, mlp: [1] } }
  expect(validateTrace(t).join()).toMatch(/candidate \d+ votes are not 3 finite numbers per block/)
})

const neuronData = (layers: number, dims = 576, neurons = 4): NeuronData => ({
  neurons, invRms: 0.05,
  activations: packHalf(Array.from({ length: layers * neurons }, (_, i) => i / 10)),
  candidates: [{ id: 100, row: packFloat32(new Float32Array(dims)) }],
})

const withNeurons = (n: unknown) => buildFixtureTrace({ mode: 'real', mlp: true }).map((e) =>
  e.type === 'edits' ? { ...e, neurons: n as NeuronData } : e)

test('a well-formed neurons field passes', () => {
  expect(validateTrace(withNeurons(neuronData(3)))).toEqual([])
})

test('neuronProblems names each defect', () => {
  expect(neuronProblems(neuronData(3), 3, 576)).toEqual([])
  expect(neuronProblems(neuronData(2), 3, 576)[0]).toMatch(/activations hold 8 values, expected 3 × 4/)
  expect(neuronProblems({ ...neuronData(3), activations: '%%' }, 3, 576)[0]).toMatch(/not base64 float16/)
  expect(neuronProblems({ ...neuronData(3), invRms: -1 }, 3, 576)[0]).toMatch(/invRms/)
  expect(neuronProblems(neuronData(3, 10), 3, 576)[0]).toMatch(/candidate 100 row is not 576 float32 values/)
  expect(neuronProblems(null, 3, 576)).toEqual(['neurons is not an object'])
})

test('neuronProblems rejects a candidate row with non-finite values', () => {
  const good = neuronData(3)
  const row = new Float32Array(576).fill(1); row[5] = NaN
  const bad = { ...good, candidates: [{ ...good.candidates[0], row: packFloat32(row) }] }
  expect(neuronProblems(bad, 3, 576)).toEqual([`candidate ${good.candidates[0].id} row is not all finite`])
})

test('validateTrace reports a broken neurons field with its cycle', () => {
  expect(validateTrace(withNeurons(neuronData(2)))[0]).toMatch(/^edits cycle 0 activations hold/)
})

test('dropBrokenNeurons removes only the broken fields and counts them', () => {
  const events = withNeurons(neuronData(3))
  const firstEdits = events.findIndex((e) => e.type === 'edits')
  events[firstEdits] = { ...(events[firstEdits] as Extract<TraceEvent, { type: 'edits' }>), neurons: neuronData(2) }
  const { events: kept, droppedNeurons, droppedSteps } = dropBrokenNeurons(events)
  expect(droppedNeurons).toBe(1)
  expect(droppedSteps).toBe(0)
  const edits = kept.filter((e): e is Extract<TraceEvent, { type: 'edits' }> => e.type === 'edits')
  expect(edits[0].neurons).toBeUndefined()
  expect(edits[1].neurons).toBeDefined()
  expect(validateTrace(kept)).toEqual([])
})

const steps = (layers: number, neurons = 4, dims = 576): MlpSteps => ({
  gate: packHalf(Array.from({ length: layers * neurons }, (_, i) => i / 7 - 1)),
  up: packHalf(Array.from({ length: layers * neurons }, (_, i) => Math.sin(i))),
  mlpIn: packHalf(Array.from({ length: layers * dims }, (_, i) => Math.cos(i))),
  mlpOut: packHalf(Array.from({ length: layers * dims }, (_, i) => Math.sin(i) / 3)),
})

test('well-formed steps pass', () => {
  expect(stepProblems(steps(3), 3, 4, 576)).toEqual([])
})

test('steps with a wrong length or non-finite values are reported by field', () => {
  expect(stepProblems({ ...steps(3), up: packHalf([1, 2, 3]) }, 3, 4, 576)).toEqual(['steps.up holds 3 values, expected 3 × 4'])
  expect(stepProblems({ ...steps(3), mlpOut: packHalf(Array(3 * 576).fill(NaN)) }, 3, 4, 576)).toEqual(['steps.mlpOut values are not all finite'])
  expect(stepProblems(null, 3, 4, 576)).toEqual(['steps is not an object'])
})

test('an import drops only broken steps and keeps the neuron data', () => {
  const events = withNeurons({ ...neuronData(3), steps: { ...steps(3), gate: 'not base64!' } })
  const { events: out, droppedNeurons, droppedSteps } = dropBrokenNeurons(events)
  const edits = out.filter((e) => e.type === 'edits') as Array<Extract<TraceEvent, { type: 'edits' }>>
  expect(droppedSteps).toBe(edits.length)
  expect(droppedNeurons).toBe(0)
  expect(edits.every((e) => e.neurons !== undefined && e.neurons.steps === undefined)).toBe(true)
  expect(validateTrace(out)).toEqual([])
})

test('whole neuron drops and steps-only drops are counted separately', () => {
  const events = withNeurons({ ...neuronData(3), steps: steps(3) })
  const at = events.flatMap((e, i) => (e.type === 'edits' ? [i] : []))
  expect(at.length).toBeGreaterThanOrEqual(2)
  const ed = (i: number) => events[i] as Extract<TraceEvent, { type: 'edits' }>
  events[at[0]] = { ...ed(at[0]), neurons: neuronData(2) }
  events[at[1]] = { ...ed(at[1]), neurons: { ...ed(at[1]).neurons!, steps: { ...steps(3), up: 'not base64!' } } }
  const { droppedNeurons, droppedSteps } = dropBrokenNeurons(events)
  expect(droppedNeurons).toBe(1)
  expect(droppedSteps).toBe(1)
})

test('tokenize tokens must each be an integer id with a text', () => {
  const at = (bad: unknown) => {
    const t = buildFixtureTrace({ chat: true })
    const k = t.findIndex((e) => e.type === 'tokenize')
    const e = t[k]
    if (e.type !== 'tokenize') throw new Error('no tokenize')
    t[k] = { ...e, tokens: [...e.tokens.slice(0, 3), bad, ...e.tokens.slice(4)] as typeof e.tokens }
    return validateTrace(t)
  }
  expect(at({ id: 7, text: 'x' })).toEqual([])
  for (const bad of [null, 'x', { id: 7 }, { text: 'x' }, { id: 1.5, text: 'x' }, { id: 7, text: 3 }])
    expect(at(bad)).toContain('tokenize token 3 is not { id, text }')
  const t = buildFixtureTrace()
  t[1] = { type: 'tokenize', tokens: 'abc' as unknown as [] }
  expect(validateTrace(t)).toContain('tokenize tokens is not a list')
})

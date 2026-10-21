import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import * as asset from '../neurons/asset'
import * as steps from '../neurons/steps'
import * as stripMath from './stripMath'
import { decodeSteps, editRatio, layerSteps } from '../neurons/steps'
import { buildFixtureTrace } from '../test/fixtures'
import { fixtureNeuronAsset, fixtureNeuronLayer, NF } from '../test/neuronFixture'
import { packHalf, unpackHalf } from '../trace/neuronPack'
import type { NeuronData } from '../trace/types'
import { MlpStepsCard } from './MlpStepsCard'
import { NeuronCard } from './NeuronCard'
import { useMlpLayer } from './mlpLayerSelection'
import { useNeuronMode } from './neuronMode'
import { cycleNeurons, cycleVotes, type VotedCandidate } from './selectors'

const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
const last = events.length - 1
const data = (cycleNeurons(events, last) as { data: NeuronData }).data
const candidates = (cycleVotes(events, last) as { candidates: VotedCandidate[] }).candidates
const run = events[0]

const card = (props: Partial<Parameters<typeof MlpStepsCard>[0]> = {}) =>
  render(<MlpStepsCard data={data} candidates={candidates} note="the note" run={run} {...props} />)

beforeEach(() => {
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  vi.spyOn(asset, 'loadNeuronLayer').mockImplementation(async (l) => fixtureNeuronLayer(l))
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

test('without step data it shows the note it is given', () => {
  card({ data: { ...data, steps: undefined } })
  expect(screen.getByTestId('steps-note')).toHaveTextContent('the note')
})

test('opens on the same layer as Inside one MLP and follows the shared stepper', async () => {
  render(<><NeuronCard data={data} candidates={candidates} note="n" run={run} /><MlpStepsCard data={data} candidates={candidates} note="n" run={run} /></>)
  const layerOf = (id: string) => screen.getByTestId(id).textContent!.match(/layer (\d+)/)![1]
  expect(layerOf('steps-title')).toBe(layerOf('neuron-title'))
  fireEvent.click(screen.getByTestId(Number(layerOf('steps-title')) > 0 ? 'steps-prev' : 'steps-next'))
  expect(layerOf('steps-title')).toBe(layerOf('neuron-title'))
  await waitFor(() => screen.getByTestId('strip-n'))
})

test('draws all five steps once the layer file is loaded', async () => {
  card()
  expect(screen.getByTestId('steps-scatter')).toBeInTheDocument()          // steps 2–4 need no layer file
  expect(screen.getAllByTestId('steps-loading')).toHaveLength(2)            // steps 1 and 5 wait for it
  await waitFor(() => expect(screen.getByTestId('strip-n')).toBeInTheDocument())
  for (const id of ['strip-x', 'strip-w', 'steps-silu', 'steps-product', 'strip-edit']) expect(screen.getByTestId(id)).toBeInTheDocument()
  expect(screen.getByTestId('steps-card')).toHaveTextContent(`+ ${NF.neurons - 3} more`)
})

test('the ratio line equals ‖edit‖ / ‖x‖ of this layer', async () => {
  card()
  await waitFor(() => screen.getByTestId('steps-ratio'))
  const layer = Number(screen.getByTestId('steps-title').textContent!.match(/layer (\d+)/)![1])
  const d = decodeSteps(data.steps!)
  const s = layerSteps(d, unpackHalf(data.activations), layer, NF.neurons, NF.dims)!
  expect(screen.getByTestId('steps-ratio')).toHaveTextContent(`${editRatio(s.edit, s.x).toFixed(2)} of the stream’s length`)
})

test('the marked neurons are the ones Inside one MLP lists, in its current mode', async () => {
  card()
  await waitFor(() => screen.getByTestId('strip-n'))
  const marked = () => [...screen.getByTestId('steps-scatter').querySelectorAll('.step-dot-marked')].map((e) => Number(e.getAttribute('data-neuron'))).sort((a, b) => a - b)
  const voteMarked = marked()
  expect(voteMarked).toHaveLength(8)
  act(() => useNeuronMode.getState().setMode('size'))
  await waitFor(() => expect(marked()).not.toEqual(voteMarked))
})

test('hovering a neuron shows its gate, SiLU, up, product and labels', async () => {
  card()
  await waitFor(() => screen.getByTestId('strip-n'))
  const dot = screen.getByTestId('steps-scatter').querySelector('.step-dot-marked')!
  fireEvent.mouseEnter(dot)
  const box = screen.getByTestId('steps-box')
  expect(box).toHaveTextContent(new RegExp(`^N${dot.getAttribute('data-neuron')}`))
  expect(box).toHaveTextContent(/g −?\d+\.\d{2} → SiLU −?\d+\.\d{2}/)
  expect(box).toHaveTextContent(/× u −?\d+\.\d{2} = h −?\d+\.\d{2}/)
  expect(box).toHaveTextContent(/pushes toward/)
})

test('the hover box opens upward for a mark in the lower half of a chart, downward otherwise', async () => {
  card()
  await waitFor(() => screen.getByTestId('strip-n'))
  const silu = screen.getByTestId('steps-silu')
  const dots = [...silu.querySelectorAll('.step-dot-marked')]
  const cy = (e: Element) => Number(e.getAttribute('cy'))
  const low = dots.find((e) => cy(e) > 75)!
  fireEvent.mouseEnter(low)
  expect(screen.getByTestId('steps-box').style.transform).toMatch(/, calc\(-100% - 8px\)\)$/)
  fireEvent.mouseLeave(low)
  const high = dots.find((e) => cy(e) <= 75)!
  fireEvent.mouseEnter(high)
  expect(screen.getByTestId('steps-box').style.transform).toMatch(/, 8px\)$/)
})

test('stepping while a layer file loads never mixes layers', async () => {
  let release!: () => void
  vi.mocked(asset.loadNeuronLayer).mockImplementation((l) => l === 0
    ? new Promise((res) => { release = () => res(fixtureNeuronLayer(0)) }) : Promise.resolve(fixtureNeuronLayer(l)))
  useMlpLayer.getState().select(run, 0)
  card()
  fireEvent.click(screen.getByTestId('steps-next'))
  await waitFor(() => screen.getByTestId('strip-w'))
  const w1 = screen.getByTestId('strip-w').innerHTML
  await act(async () => { release(); await new Promise((r) => setTimeout(r, 0)) })
  expect(screen.getByTestId('steps-title')).toHaveTextContent('layer 1')
  expect(screen.getByTestId('strip-w').innerHTML).toBe(w1)
})

test('a load error offers retry', async () => {
  vi.mocked(asset.loadNeuronLayer).mockRejectedValueOnce(new Error('HTTP 503'))
  card()
  await waitFor(() => expect(screen.getAllByTestId('steps-error')).toHaveLength(2))
  fireEvent.click(screen.getAllByTestId('steps-retry')[0])
  await waitFor(() => screen.getByTestId('strip-n'))
})

test('a weights file of another size shows the mismatch note in steps 1 and 5 instead of loading forever', async () => {
  const a = fixtureNeuronAsset()
  vi.mocked(asset.loadNeuronAsset).mockResolvedValue({ ...a, manifest: { ...a.manifest, neurons: a.manifest.neurons + 1 } })
  card()
  await waitFor(() => expect(screen.getAllByTestId('steps-mismatch')).toHaveLength(2))
  expect(screen.getAllByTestId('steps-mismatch')[0]).toHaveTextContent('This run’s step data doesn’t match the MLP weights file.')
  expect(screen.queryByTestId('steps-loading')).toBeNull()
  expect(screen.getByTestId('steps-scatter')).toBeInTheDocument()
})

const INCONSISTENT = 'This run’s step data is inconsistent and can’t be shown.'

test('step data whose sizes disagree shows the inconsistent-data note', () => {
  const d = decodeSteps(data.steps!)
  card({ data: { ...data, steps: { ...data.steps!, gate: packHalf(d.gate.subarray(1)) } } })
  expect(screen.getByTestId('steps-note')).toHaveTextContent(INCONSISTENT)
})

test('step data with a fractional number of dimensions shows the inconsistent-data note', () => {
  const d = decodeSteps(data.steps!)
  const longer = (a: Float32Array) => packHalf([...a, 0])
  card({ data: { ...data, steps: { ...data.steps!, mlpIn: longer(d.mlpIn), mlpOut: longer(d.mlpOut) } } })
  expect(screen.getByTestId('steps-note')).toHaveTextContent(INCONSISTENT)
})

test('a hover redraws only the box: no summary is recomputed and no strip re-renders', async () => {
  card()
  await waitFor(() => screen.getByTestId('strip-edit'))
  const spies = [vi.spyOn(steps, 'normalize'), vi.spyOn(steps, 'productSummary'), vi.spyOn(steps, 'gateDomain'),
    vi.spyOn(steps, 'gateCounts'), vi.spyOn(stripMath, 'cellFill')]
  const dot = screen.getByTestId('steps-scatter').querySelector('.step-dot-marked')!
  fireEvent.mouseEnter(dot)
  expect(screen.getByTestId('steps-box')).toBeInTheDocument()
  fireEvent.mouseLeave(dot)
  for (const spy of spies) expect(spy).not.toHaveBeenCalled()
})

test('the norm note says x and n are clipped and w is scaled to its own peak', async () => {
  card()
  await waitFor(() => screen.getByTestId('strip-n'))
  expect(screen.getByTestId('steps-card')).toHaveTextContent('x and n clipped at 2× their typical size, w scaled to its largest value.')
})

test('each ⓘ chip hovers the sentence of the section it opens', async () => {
  card()
  await waitFor(() => screen.getAllByTestId('steps-callout'))
  const tips = screen.getAllByTestId('steps-callout').map((chip) => {
    fireEvent.mouseEnter(chip)
    const text = screen.getByTestId('gl-tip').textContent
    fireEvent.mouseLeave(chip)
    return text
  })
  expect(tips[0]).toContain('SiLU is smooth')
  expect(tips[1]).toContain('The gate decides how much passes')
  expect(tips[2]).toContain('about a hundred neurons')
})

test('a note under the scatter says which dots carry numbers and how to see the others', async () => {
  card()
  await waitFor(() => screen.getByTestId('steps-scatter'))
  expect(screen.getByTestId('steps-scatter-note')).toHaveTextContent(
    'Numbers: the three marked neurons with the largest activations. Hover or focus any dot for its number.')
})

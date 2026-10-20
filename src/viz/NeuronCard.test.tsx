import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import * as asset from '../neurons/asset'
import * as math from '../neurons/math'
import { buildFixtureTrace } from '../test/fixtures'
import { fixtureNeuronAsset, fixtureNeuronLayer, NF } from '../test/neuronFixture'
import type { NeuronData, TraceEvent } from '../trace/types'
import { NeuronCard } from './NeuronCard'
import { cycleNeurons, cycleVotes, type VotedCandidate } from './selectors'
import { useMlpLayer } from './mlpLayerSelection'

const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
const last = events.length - 1
const data = (cycleNeurons(events, last) as { data: NeuronData }).data
const candidates = (cycleVotes(events, last) as { candidates: VotedCandidate[] }).candidates
const run = events[0]

const card = (props: Partial<Parameters<typeof NeuronCard>[0]> = {}) =>
  render(<NeuronCard data={data} candidates={candidates} note="the note" run={run} {...props} />)

beforeEach(() => {
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  vi.spyOn(asset, 'loadNeuronLayer').mockImplementation(async (l) => fixtureNeuronLayer(l))
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

const largestLayer = (c: VotedCandidate) => c.votes.mlp.reduce((best, v, l, a) => (Math.abs(v) > Math.abs(a[best]) ? l : best), 0)

test('without data it shows the note it is given', () => {
  card({ data: undefined })
  expect(screen.getByTestId('neuron-note')).toHaveTextContent('the note')
})

test('opens on the layer with the largest MLP vote, shows loading, then 8 rows that add up to the votes card', async () => {
  card()
  const layer = largestLayer(candidates[0])
  expect(screen.getByTestId('neuron-title')).toHaveTextContent(`Inside one MLP: layer ${layer}`)
  expect(screen.getByTestId('neuron-loading')).toHaveTextContent('Loading the MLP weights (about 0.9 MB per layer, once per model version)…')
  await waitFor(() => expect(screen.getAllByTestId('neuron-row')).toHaveLength(8))
  expect(screen.getByTestId('neuron-rest')).toHaveTextContent(`the other ${NF.neurons - 8} neurons`)
  const cell = candidates[0].votes.mlp[layer]
  expect(screen.getByTestId('neuron-total')).toHaveTextContent(`${cell >= 0 ? '+' : '−'}${Math.abs(cell).toFixed(2)}`)
  expect(screen.getByTestId('neuron-total')).toHaveTextContent('= the votes card’s cell')
  expect(screen.queryByTestId('neuron-approx-note')).toBeNull()
})

test('labels show as visible tokens, negative votes get an away chip and an (away) badge', async () => {
  card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  expect(screen.getAllByTestId('neuron-label')[0].textContent).toMatch(/^(w\d+|␣and)$/)
  for (const r of screen.getAllByTestId('neuron-row')) {
    const negative = r.querySelector('[data-testid="neuron-vote"]')!.textContent!.startsWith('−')
    expect(!!r.querySelector('[data-testid="neuron-away"]')).toBe(negative)
    expect(r.querySelector('[data-testid="neuron-badge"]')!.textContent!.endsWith('(away)')).toBe(negative)
  }
})

test('the stepper moves one layer and stops at the ends', async () => {
  useMlpLayer.getState().select(run, 0)
  card()
  expect(screen.getByTestId('neuron-prev')).toBeDisabled()
  fireEvent.click(screen.getByTestId('neuron-next'))
  expect(screen.getByTestId('neuron-title')).toHaveTextContent('layer 1')
  fireEvent.click(screen.getByTestId('neuron-next'))
  expect(screen.getByTestId('neuron-next')).toBeDisabled()
})

test('a layer picked in another card opens here', () => {
  useMlpLayer.getState().select(run, 1)
  card()
  expect(screen.getByTestId('neuron-title')).toHaveTextContent('layer 1')
})

test('firing most drops the candidate, the badge and the total', async () => {
  card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  fireEvent.click(screen.getAllByTestId('neuron-mode-toggle')[1])
  expect(screen.getByTestId('neuron-table')).toHaveTextContent('edit size')
  expect(screen.queryByTestId('neuron-badge')).toBeNull()
  expect(screen.queryByTestId('neuron-total')).toBeNull()
  expect(screen.queryByTestId('neuron-cand')).toBeNull()
})

test('the mode survives the card remounting, e.g. after switching runs', async () => {
  const first = card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  fireEvent.click(screen.getAllByTestId('neuron-mode-toggle')[1])
  first.unmount()
  card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  expect(screen.getAllByTestId('neuron-mode-toggle')[1]).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByTestId('neuron-table')).toHaveTextContent('edit size')
})

test('picking another candidate changes the vote column', async () => {
  card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  fireEvent.click(screen.getAllByTestId('neuron-cand')[1])
  expect(screen.getByTestId('neuron-table')).toHaveTextContent(`vote for “${candidates[1].text.replace(/^ /, '␣')}”`)
})

test('a total off by more than the tolerance shows ≈ and the note', async () => {
  const off = candidates.map((c, i) => i === 0 ? { ...c, votes: { ...c.votes, mlp: c.votes.mlp.map((v) => v + 5) } } : c)
  card({ candidates: off })
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  expect(screen.getByTestId('neuron-total')).toHaveTextContent('≈')
  expect(screen.getByTestId('neuron-approx-note')).toHaveTextContent('The browser’s 8-bit model rounds this MLP’s inputs')
})

test('a load error offers retry', async () => {
  vi.mocked(asset.loadNeuronLayer).mockRejectedValueOnce(new Error('HTTP 503'))
  card()
  await waitFor(() => expect(screen.getByTestId('neuron-error')).toHaveTextContent('Couldn’t load the MLP weights.'))
  fireEvent.click(screen.getByTestId('neuron-retry'))
  await waitFor(() => expect(screen.getAllByTestId('neuron-row')).toHaveLength(8))
})

test('neuron data that does not match the weights file shows a note instead of throwing', async () => {
  card({ data: { ...data, neurons: 7 } })
  await waitFor(() => expect(screen.getByTestId('neuron-note')).toHaveTextContent('This run’s neuron data doesn’t match the MLP weights file.'))
})

test('a candidate without a row shows a note instead of throwing', async () => {
  card({ data: { ...data, candidates: data.candidates.slice(1) } })
  await waitFor(() => expect(screen.getByTestId('neuron-note')).toHaveTextContent('No neuron data for this candidate.'))
})

test('the run-start of another trace resets a pick', () => {
  useMlpLayer.getState().select(buildFixtureTrace()[0] as TraceEvent, 2)
  card()
  expect(screen.getByTestId('neuron-title')).toHaveTextContent(`layer ${largestLayer(candidates[0])}`)
})

test('the vote table is computed once per (run, cycle, layer, candidate), not per render', async () => {
  const spy = vi.spyOn(math, 'behindTheVote')
  const { rerender } = card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  const calls = spy.mock.calls.length
  rerender(<NeuronCard data={data} candidates={candidates} note="the note" run={run} />)
  expect(screen.getAllByTestId('neuron-row')).toHaveLength(8)
  expect(spy.mock.calls.length).toBe(calls)
})

test('firing most is computed once per layer, not per render', async () => {
  const spy = vi.spyOn(math, 'firingMost')
  const { rerender } = card()
  await waitFor(() => screen.getAllByTestId('neuron-row'))
  fireEvent.click(screen.getAllByTestId('neuron-mode-toggle')[1])
  const calls = spy.mock.calls.length
  expect(calls).toBeGreaterThan(0)
  rerender(<NeuronCard data={data} candidates={candidates} note="the note" run={run} />)
  expect(spy.mock.calls.length).toBe(calls)
})

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import type { KnockoutLayer, KnockoutSweep, TraceEvent } from '../trace/types'
import { KnockoutCard } from './KnockoutCard'
import { useKnockoutRequest } from './knockoutRequest'
import { useMlpLayer } from './mlpLayerSelection'

afterEach(cleanup)

type Run = Extract<TraceEvent, { type: 'run-start' }>
const base: Run = { type: 'run-start', prompt: 'The capital of France is', mode: 'real', modelId: 'm',
  params: { temperature: 0.8, topK: 10, maxNewTokens: 4 }, seed: 7, revision: 'v7' }
const model = { knockout: true, revision: 'v7' }
const paris = { id: 5, text: ' Paris' }
const the = { id: 6, text: ' the' }

const layer = (l: number, prob: number, flips = false): KnockoutLayer =>
  ({ layer: l, prob, top: flips ? { ...the, prob: 0.5 } : { ...paris, prob }, flips })
const sweep: KnockoutSweep = { cycle: 2, revision: 'v7', chosen: paris, baseline: 0.98,
  layers: Array.from({ length: 30 }, (_, l) => l === 0 ? layer(0, 0.02, true) : l === 15 ? layer(15, 0.4) : l === 29 ? layer(29, 1) : layer(l, 0.97)) }
const votes = Array.from({ length: 30 }, (_, l) => (l === 15 ? 0.1 : l === 27 ? 2 : 0.3))

function card(props: Partial<React.ComponentProps<typeof KnockoutCard>> = {}) {
  return render(<KnockoutCard run={base} cycle={2} seqLen={10} chosen={paris} votes={votes} sweep={undefined}
    running={null} model={model} recordId="r1" {...props} />)
}
const note = () => screen.getByTestId('knockout-note').textContent

test('title is exact', () => {
  card()
  expect(screen.getByRole('heading', { name: /Which layers the chosen token depends on/ })).toBeInTheDocument()
})

test.each([
  [{ run: { ...base, mode: 'sim' as const } }, 'Real model only: simulated runs have no MLP to measure.'],
  [{ model: null }, 'Load the real model to measure knockouts.'],
  [{ model: { knockout: false, revision: 'v7' } }, 'Load the real model to measure knockouts.'],
  [{ run: { ...base, seed: undefined } }, 'Recorded with an earlier version of the model — generate the run again to measure knockouts.'],
  [{ model: { knockout: true, revision: 'v8' } }, 'Recorded with an earlier version of the model — generate the run again to measure knockouts.'],
  [{ run: { ...base, knockout: { layer: 15, cycle: 1, sourceId: 'x' } } }, 'This run already has MLP 15 removed — open the original run to measure.'],
  [{ seqLen: 513 }, 'This run is longer than 512 tokens, too long to measure.'],
  [{ chosen: undefined }, 'Measure once the run has ended.'],
])('note for %#', (props, text) => {
  card({ ...props, noteAction: <button data-testid="act">go</button> })
  expect(note()).toContain(text)
  expect(screen.queryByText('Measure all 30 layers')).toBeNull()
})

test('the load-real action appears for sim and no-model notes only', () => {
  card({ model: null, noteAction: <button data-testid="act">go</button> })
  expect(screen.getByTestId('act')).toBeInTheDocument()
  cleanup()
  card({ seqLen: 600, noteAction: <button data-testid="act">go</button> })
  expect(screen.queryByTestId('act')).toBeNull()
})

test('eligible without a sweep: the button calls measure with the cycle', () => {
  const measure = vi.fn()
  useKnockoutRequest.setState({ measure })
  card()
  fireEvent.click(screen.getByRole('button', { name: 'Measure all 30 layers' }))
  expect(measure).toHaveBeenCalledWith(2)
})

test('the button is disabled until the app installs measure', () => {
  card()
  expect(screen.getByRole('button', { name: 'Measure all 30 layers' })).toBeDisabled()
})

test('a sweep running for this cycle fills bars as they arrive and disables the button', () => {
  useKnockoutRequest.setState({ measure: vi.fn() })
  card({ running: { recordId: 'r1', cycle: 2, baseline: 0.98, layers: [layer(0, 0.5), layer(1, 0.9)] } })
  expect(screen.getAllByTestId('knockout-bar')).toHaveLength(2)
  expect(screen.getByRole('button', { name: 'Measure all 30 layers' })).toBeDisabled()
})

test('a sweep running for another record or cycle only disables the button', () => {
  useKnockoutRequest.setState({ measure: vi.fn() })
  card({ running: { recordId: 'other', cycle: 2, baseline: 0.98, layers: [layer(0, 0.5)] } })
  expect(screen.queryAllByTestId('knockout-bar')).toHaveLength(0)
  expect(screen.getByRole('button', { name: 'Measure all 30 layers' })).toBeDisabled()
  cleanup()
  card({ running: { recordId: 'r1', cycle: 3, baseline: 0.98, layers: [layer(0, 0.5)] } })
  expect(screen.queryAllByTestId('knockout-bar')).toHaveLength(0)
  expect(screen.getByRole('button', { name: 'Measure all 30 layers' })).toBeDisabled()
})

test('a saved sweep shows 30 bars, drop values, flips, vote row and role chips', () => {
  card({ sweep })
  const bars = screen.getAllByTestId('knockout-bar')
  expect(bars).toHaveLength(30)
  expect(bars[15]).toHaveAttribute('data-drop', '0.58')
  expect(bars[29].getAttribute('data-drop')).toBe('-0.02')
  expect(bars[0]).toHaveAttribute('data-flips', 'true')
  expect(bars[1]).toHaveAttribute('data-flips', 'false')
  expect(bars[15]).toHaveAccessibleName('MLP 15: “␣Paris” 98% → 40%')
  expect(screen.getAllByTestId('knockout-vote')).toHaveLength(30)
  expect(screen.queryByText('Measure all 30 layers')).toBeNull()
  const chips = screen.getAllByTestId('knockout-chip').map((c) => c.textContent)
  expect(chips[0]).toBe('MLP 0 · decisive · “␣Paris” → “␣the”')
  expect(chips).toContain('MLP 15 · preparatory · −58 pts')
})

test('clicking a bar shows the detail line, the action, and sets the shared layer', () => {
  const runKnockout = vi.fn()
  useKnockoutRequest.setState({ runKnockout })
  card({ sweep })
  fireEvent.click(screen.getAllByTestId('knockout-bar')[15])
  expect(screen.getByTestId('knockout-detail')).toHaveTextContent('MLP 15 — “␣Paris” 98% → 40%, top token unchanged')
  expect(useMlpLayer.getState().pick).toEqual({ run: base, layer: 15 })
  fireEvent.click(screen.getByRole('button', { name: 'Run with MLP 15 off and compare' }))
  expect(runKnockout).toHaveBeenCalledWith(15, 2)
})

test('a flipping layer names the new top token; a chip selects its layer', () => {
  card({ sweep })
  fireEvent.click(screen.getAllByTestId('knockout-chip')[0])
  expect(screen.getByTestId('knockout-detail')).toHaveTextContent('MLP 0 — “␣Paris” 98% → 2%, top token becomes “␣the”')
  expect(useMlpLayer.getState().pick?.layer).toBe(0)
})

test('the action is disabled while a sweep runs', () => {
  useKnockoutRequest.setState({ runKnockout: vi.fn(), running: { recordId: 'r1', cycle: 2, baseline: 0.98, layers: sweep.layers } })
  card({ sweep: undefined, running: useKnockoutRequest.getState().running })
  fireEvent.click(screen.getAllByTestId('knockout-bar')[15])
  expect(screen.getByRole('button', { name: 'Run with MLP 15 off and compare' })).toBeDisabled()
})

test('chips show the change in points: rises with a plus, compensated too', () => {
  const layers = sweep.layers.map((l) => l.layer === 27 ? layer(27, 0.84) : l.layer === 28 ? layer(28, 1) : layer(l.layer, 0.85))
  card({ sweep: { ...sweep, baseline: 0.85, layers } })
  const chips = screen.getAllByTestId('knockout-chip').map((c) => c.textContent)
  expect(chips).toContain('MLP 27 · compensated · −1 pts')
  expect(chips).toContain('MLP 28 · dampening · +15 pts')
})

test('the detail wording follows the top token, not flips', () => {
  const layers = sweep.layers.map((l) => l.layer === 3 ? { ...layer(3, 0.5), top: { ...the, prob: 0.6 } }
    : l.layer === 4 ? { ...layer(4, 0.5), flips: true } : l)
  card({ sweep: { ...sweep, layers } })
  fireEvent.click(screen.getAllByTestId('knockout-bar')[3])
  expect(screen.getByTestId('knockout-detail')).toHaveTextContent('top token becomes “␣the”')
  fireEvent.click(screen.getAllByTestId('knockout-bar')[4])
  expect(screen.getByTestId('knockout-detail')).toHaveTextContent('98% → 50%, top token unchanged')
})

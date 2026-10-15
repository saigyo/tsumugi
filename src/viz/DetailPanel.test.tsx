import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace, makeFixtureTrace, makeGridEvent } from '../test/fixtures'
import type { AttentionHead } from '../trace/types'
import { DetailPanel } from './DetailPanel'
import { texOf } from '../test/tex'

const trace = makeFixtureTrace()  // cursor 1=tokenize, 2=embed, 3..5=layers, 6=attention

afterEach(() => cleanup())

// Mock getContext to prevent jsdom stub warnings in test output (the grid
// explorer's canvas thumbnails), same as AttentionGridExplorer.test.tsx.
vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as never)

vi.mock('../geometry/useGeometry', () => ({ useGeometry: () => ({ status: 'loading', retry: () => {} }) }))

test('tokenizer detail shows token chips with ids', () => {
  render(<DetailPanel events={trace} cursor={1} mode="sim" />)
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('The')
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('10')  // token id
  // ' cat' carries a leading space: shown as a marker, and explained once
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('␣cat')
  expect(screen.getByTestId('tokenizer-space-note')).toHaveTextContent(/leading space/)
})

test('embeddings detail shows the embedding-table shape', () => {
  render(<DetailPanel events={trace} cursor={2} mode="sim" />)
  expect(screen.getByTestId('detail-embeddings')).toHaveTextContent('49 152 × 576')
  expect(screen.getAllByTestId('embed-token')).toHaveLength(2)
})

test('layers detail lights the active layer; sim shows norms', () => {
  render(<DetailPanel events={trace} cursor={4} mode="sim" />)  // layer index 1 of 3
  const blocks = screen.getAllByTestId('layer-block')
  expect(blocks).toHaveLength(3)
  expect(blocks[1].dataset.lit).toBe('true')
  expect(blocks[2].dataset.lit).toBe('false')
})

test('real mode labels layers as schematic when the run has no attention data', () => {
  // The schematic tag is a run-level fact, not per-cycle: it should show whenever
  // this run never produces real attention data anywhere, not just before the
  // current cycle's attention event has fired (see the cursor-6/cursor-4-with-
  // attention cases below).
  const noAttn = trace.filter((e) => e.type !== 'attention')
  render(<DetailPanel events={noAttn} cursor={4} mode="real" />)
  expect(screen.getByTestId('detail-layers')).toHaveTextContent(/schematic/i)
})

test('real mode drops the schematic tag pre-emptively when the run will produce attention data', () => {
  // cursor 4 is a layer event in cycle 0, before cycle 0's own attention event
  // (cursor 6) — but the full run does produce attention data later, so the tag
  // must not appear even this early.
  render(<DetailPanel events={trace} cursor={4} mode="real" />)
  expect(screen.getByTestId('detail-layers')).not.toHaveTextContent(/schematic/i)
})

test('no relevant event renders empty state', () => {
  render(<DetailPanel events={trace} cursor={-1} mode="sim" />)
  expect(screen.getByTestId('detail-empty')).toBeInTheDocument()
})

test('truncated tokenize event shows a notice', () => {
  const t = makeFixtureTrace()
  const tok = t[1]
  if (tok.type === 'tokenize') tok.truncated = true
  render(<DetailPanel events={t} cursor={1} mode="sim" />)
  expect(screen.getByTestId('truncation-notice')).toBeInTheDocument()
})

// makeFixtureTrace cycle 0: index 7=logits, 8=softmax, 9=sample, 10=append
test('logits detail shows one bar per candidate', () => {
  render(<DetailPanel events={trace} cursor={7} mode="sim" />)
  expect(screen.getAllByTestId('logit-bar')).toHaveLength(3)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('sat')
})

test('softmax cursor shows the Softmax card with probabilities', () => {
  render(<DetailPanel events={trace} cursor={8} mode="sim" />)
  expect(screen.queryByTestId('detail-logits')).toBeNull()
  expect(screen.getByTestId('detail-softmax')).toHaveTextContent('70%')  // 0.7 → 70%
})

test('sampler detail marks the chosen token', () => {
  render(<DetailPanel events={trace} cursor={9} mode="sim" />)
  expect(screen.getByTestId('chosen-marker')).toHaveTextContent('sat')
})

test('run-end cursor shows a run summary, not the idle hint', () => {
  render(<DetailPanel events={trace} cursor={trace.length - 1} mode="sim" />)
  const summary = screen.getByTestId('detail-run-end')
  expect(summary).toHaveTextContent(/max-tokens/)
  expect(summary).toHaveTextContent('2')          // generated token count
  expect(summary).toHaveTextContent(/T=0.8/)      // params from run-start
  expect(screen.queryByTestId('detail-empty')).not.toBeInTheDocument()
})

test('attention cursor maps to layers stage and shows the heatmap', () => {
  render(<DetailPanel events={trace} cursor={6} mode="sim" />)
  expect(screen.getByTestId('detail-layers')).toBeInTheDocument()
  expect(screen.getByTestId('attention-heatmap')).toBeInTheDocument()
})

test('real mode drops the schematic tag when attention data exists', () => {
  render(<DetailPanel events={trace} cursor={6} mode="real" />)  // cursor on attention event
  expect(screen.getByTestId('detail-layers')).not.toHaveTextContent(/schematic/i)
})

test('mid-layer cursor before the cycle attention has no heatmap', () => {
  render(<DetailPanel events={trace} cursor={4} mode="sim" />)
  expect(screen.queryByTestId('attention-heatmap')).not.toBeInTheDocument()
})

test('layers detail shows the residual-stream diagram with live shapes', () => {
  render(<DetailPanel events={trace} cursor={4} mode="sim" />)
  const diagram = screen.getByTestId('residual-diagram')
  expect(diagram).toHaveTextContent('[2×576]')
  expect(diagram).toHaveTextContent('× 3 layers')
  expect(diagram).toHaveTextContent(/attention/i)
  expect(diagram).toHaveTextContent(/MLP/i)
  expect(screen.getByTestId('rs-term-mlp')).toHaveAttribute('role', 'button')  // the MLP box opens its glossary entry
})

test('logits detail frames scores as a dot-product readout', () => {
  render(<DetailPanel events={trace} cursor={7} mode="sim" />)
  const formula = screen.getByTestId('logits-formula')
  expect(texOf(formula)).toContain('[1 \\times 576]')
  expect(texOf(formula)).toContain('[576 \\times 49\\,152]')
  expect(screen.getByTestId('detail-logits')).toHaveTextContent(/dot product/i)
})

function traceWithGrid() {
  const t = makeFixtureTrace()
  t.splice(t.length - 1, 0, makeGridEvent(2, 2))
  return t
}

test('layers detail offers the explorer toggle when the run has a grid', () => {
  render(<DetailPanel events={traceWithGrid()} cursor={3} mode="real" />)
  const toggle = screen.getByTestId('btn-explore-heads')
  expect(toggle).toHaveTextContent('Explore all heads (4)')
  fireEvent.click(toggle)
  expect(screen.getByTestId('grid-explorer')).toBeInTheDocument()
})

test('no explorer toggle without a grid event', () => {
  render(<DetailPanel events={trace} cursor={3} mode="sim" />)
  expect(screen.queryByTestId('btn-explore-heads')).toBeNull()
})

test('cell clicks reach onPin and pinned heads render as chips', () => {
  const onPin = vi.fn()
  const pinned: AttentionHead[] = [{ layer: 1, head: 1, label: 'pinned', matrix: [[1], [0.5, 0.5]] }]
  render(<DetailPanel events={traceWithGrid()} cursor={6} mode="real"
    pinnedHeads={pinned} onPin={onPin} pinNote={null} />)
  fireEvent.click(screen.getByTestId('btn-explore-heads'))
  fireEvent.click(screen.getAllByTestId('grid-cell')[0])
  expect(onPin).toHaveBeenCalledWith(0, 0)
  const chips = screen.getAllByTestId('head-chip')
  expect(chips.some((c) => c.textContent?.includes('pinned'))).toBe(true)
})

test('the last pinned head is focused (selected) in the viewer', () => {
  const pinned: AttentionHead[] = [
    { layer: 1, head: 1, label: 'pinned', matrix: [[1], [0.5, 0.5]] },
    { layer: 2, head: 2, label: 'pinned', matrix: [[1], [0.4, 0.6]] },
  ]
  render(<DetailPanel events={trace} cursor={6} mode="real" pinnedHeads={pinned} />)
  const chips = screen.getAllByTestId('head-chip')
  const lastPinChip = chips.find((c) => c.textContent?.includes('L2') && c.textContent?.includes('H2'))
  expect(lastPinChip).toBeDefined()
  expect(lastPinChip!.dataset.active).toBe('true')
  const otherPinChip = chips.find((c) => c.textContent?.includes('L1') && c.textContent?.includes('H1'))
  expect(otherPinChip!.dataset.active).toBe('false')
})

test('pinned-head rows are labeled with run tokens, not indices', () => {
  // cursor 3: a layer event before the cycle's attention event, so the cycle
  // contributes no token sizing (rows = 0) — labels must come from the run
  const pinned: AttentionHead[] = [{ layer: 0, head: 0, label: 'pinned', matrix: [[1], [0.5, 0.5]] }]
  render(<DetailPanel events={trace} cursor={3} mode="real" pinnedHeads={pinned} />)
  expect(screen.getAllByText('The').length).toBeGreaterThan(0)
  expect(screen.getAllByText('cat').length).toBeGreaterThan(0)
  expect(screen.queryByText('#0')).toBeNull()
})

test('the stale-pin note renders', () => {
  render(<DetailPanel events={traceWithGrid()} cursor={3} mode="real"
    pinNote="run data no longer available — regenerate to explore heads" />)
  expect(screen.getByTestId('pin-note')).toHaveTextContent('regenerate')
})

test('every stage card heading has an ⓘ that opens its entry, and captions link their first term', () => {
  const events = makeFixtureTrace()
  // one cursor per stage: tokenize 1, embed 2, layer 3, logits 7, softmax 8, sample 9
  const cases: Array<[number, string, string]> = [
    [1, 'token', 'token'], [2, 'embedding', 'residual-stream'], [3, 'layer', 'layer'],
    [7, 'logit', 'logit'], [8, 'softmax', 'logit'], [9, 'sampling', 'sampling'],
  ]
  for (const [cursor, headingEntry, captionEntry] of cases) {
    render(<DetailPanel events={events} cursor={cursor} mode="sim" />)
    const info = screen.getByTestId('stage-info')
    expect(info.closest('h3'), `cursor ${cursor}`).not.toBeNull()
    expect(info.dataset.term, `cursor ${cursor}`).toBe(headingEntry)
    expect(screen.getAllByTestId('gl-term').map((t) => t.dataset.term), `cursor ${cursor}`).toContain(captionEntry)
    cleanup()
  }
})

test('real run with MLP data: both cards below the attention section', () => {
  render(<DetailPanel events={buildFixtureTrace({ mlp: true, mode: 'real' })} cursor={4} mode="real" />)
  const layers = screen.getByTestId('detail-layers')
  expect(within(layers).getAllByTestId('edit-cell').length).toBeGreaterThan(0)
  expect(within(layers).getAllByTestId('vote-cell').length).toBeGreaterThan(0)
  const order = [...layers.querySelectorAll('[data-testid="edit-size-card"], [data-testid="votes-card"]')]
    .map((el) => el.getAttribute('data-testid'))
  expect(order).toEqual(['edit-size-card', 'votes-card'])
})

test('simulated run: both cards show the real-model-only note', () => {
  render(<DetailPanel events={buildFixtureTrace({ mlp: true })} cursor={4} mode="sim" />)
  expect(screen.getByTestId('edit-size-note')).toHaveTextContent('Real model only: simulated runs have no MLP to measure.')
  expect(screen.getByTestId('votes-note')).toHaveTextContent('Real model only: simulated runs have no MLP to measure.')
})

test('real run recorded before MLP data: the older-data note on both cards', () => {
  render(<DetailPanel events={buildFixtureTrace({ mode: 'real' })} cursor={4} mode="real" />)
  expect(screen.getByTestId('edit-size-note')).toHaveTextContent('This run was recorded before MLP data was exported.')
  expect(screen.getByTestId('votes-note')).toHaveTextContent('This run was recorded before MLP data was exported.')
})

test('live run before the cycle’s logits: the votes card says votes are coming', () => {
  const t = buildFixtureTrace({ mlp: true, mode: 'real' }).slice(0, 8)   // through cycle 0's edits
  render(<DetailPanel events={t} cursor={4} mode="real" />)
  expect(screen.getByTestId('votes-note')).toHaveTextContent('Votes appear once this cycle’s scores are computed.')
  expect(screen.getAllByTestId('edit-cell').length).toBeGreaterThan(0)
})

test('live run before the cycle’s edits: the edit-size card says they are coming, not that the run is old', () => {
  const t = buildFixtureTrace({ mlp: true, mode: 'real' }).slice(0, 7)   // through cycle 0's attention
  render(<DetailPanel events={t} cursor={4} mode="real" />)
  expect(screen.getByTestId('edit-size-note')).toHaveTextContent('Edit sizes appear once this cycle’s layers are measured.')
})

test('an ended run without the cycle’s scores shows the older-data note, not a pending one', () => {
  const t = [...buildFixtureTrace({ mlp: true, mode: 'real' }).slice(0, 8), { type: 'run-end', reason: 'error' } as const]
  render(<DetailPanel events={t} cursor={4} mode="real" />)
  expect(screen.getByTestId('votes-note')).toHaveTextContent('This run was recorded before MLP data was exported.')
})

test('real mode: the Layers detail shows the neuron card under the two MLP cards; sim mode shows its note', () => {
  const real = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
  const at = real.findIndex((e) => e.type === 'layer')
  const { unmount } = render(<DetailPanel events={real} cursor={at} mode="real" />)
  const cards = screen.getByTestId('mlp-cards').querySelectorAll('section')
  expect([...cards].map((s) => s.dataset.testid)).toEqual(['edit-size-card', 'votes-card', 'neuron-card'])
  unmount()
  render(<DetailPanel events={real} cursor={at} mode="sim" />)
  expect(screen.getByTestId('neuron-note')).toHaveTextContent('Real model only: simulated runs have no neurons to show.')
})

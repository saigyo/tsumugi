import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { makeFixtureTrace, makeGridEvent } from '../test/fixtures'
import type { AttentionHead } from '../trace/types'
import { DetailPanel } from './DetailPanel'

const trace = makeFixtureTrace()  // cursor 1=tokenize, 2=embed, 3..5=layers, 6=attention

afterEach(() => cleanup())

// Mock getContext to prevent jsdom stub warnings in test output (the grid
// explorer's canvas thumbnails), same as AttentionGridExplorer.test.tsx.
vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as never)

test('tokenizer detail shows token chips with ids', () => {
  render(<DetailPanel events={trace} cursor={1} mode="sim" />)
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('The')
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('10')  // token id
})

test('embeddings detail shows dims caption and heat cells', () => {
  render(<DetailPanel events={trace} cursor={2} mode="sim" />)
  expect(screen.getByTestId('detail-embeddings')).toHaveTextContent('576')
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

test('softmax cursor switches bars to probabilities', () => {
  render(<DetailPanel events={trace} cursor={8} mode="sim" />)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('70')  // 0.7 → 70%
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
  expect(diagram).toHaveTextContent(/multi-layer perceptron/i)  // tooltip on the MLP box
})

test('logits detail frames scores as a dot-product readout', () => {
  render(<DetailPanel events={trace} cursor={7} mode="sim" />)
  const formula = screen.getByTestId('logits-formula')
  expect(formula).toHaveTextContent('[1×576]')
  expect(formula).toHaveTextContent('[576×49 152]')
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

test('the stale-pin note renders', () => {
  render(<DetailPanel events={traceWithGrid()} cursor={3} mode="real"
    pinNote="run data no longer available — regenerate to explore heads" />)
  expect(screen.getByTestId('pin-note')).toHaveTextContent('regenerate')
})

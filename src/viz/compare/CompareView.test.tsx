import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { makeGridEvent, makeRunRecord } from '../../test/fixtures'
import type { RunRecord } from '../../app/runsStore'
import { CompareView } from './CompareView'

vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as never)

afterEach(() => cleanup())

const divergentB = () => makeRunRecord(2, { chosen: [{ id: 100, text: ' sat' }, { id: 999, text: ' off' }] })

test('identical runs: no badge, identical-outputs note, no fork marks', () => {
  render(<CompareView a={makeRunRecord(1)} b={makeRunRecord(2)} />)
  expect(screen.queryByTestId('cmp-badge')).toBeNull()
  expect(screen.getByTestId('cmp-note')).toHaveTextContent('identical outputs')
  expect(screen.getAllByTestId('cmp-token').every((t) => t.dataset.fork === 'false')).toBe(true)
})

test('metadata rows highlight only differing fields', () => {
  render(<CompareView a={makeRunRecord(1)} b={makeRunRecord(2, { temperature: 0.2 })} />)
  const rows = screen.getAllByTestId('cmp-meta-row')
  const tRow = rows.find((r) => r.textContent?.startsWith('T'))
  const promptRow = rows.find((r) => r.textContent?.startsWith('prompt'))
  expect(tRow?.dataset.diff).toBe('true')
  expect(promptRow?.dataset.diff).toBe('false')
})

test('fork cycle is marked on both streams and the ruler', () => {
  render(<CompareView a={makeRunRecord(1)} b={divergentB()} />)
  const forked = screen.getAllByTestId('cmp-token').filter((t) => t.dataset.fork === 'true')
  expect(forked).toHaveLength(2)              // one per stream, at cycle 1
  expect(forked.map((t) => t.textContent)).toEqual([' on', ' off'])
  const ticks = screen.getAllByTestId('cmp-tick')
  expect(ticks[1].dataset.fork).toBe('true')
  expect(screen.queryByTestId('cmp-note')).toBeNull()
})

test('different prompts show the badge instead of a fork', () => {
  const b = makeRunRecord(2, { prompt: 'A dog', promptTokens: [{ id: 20, text: 'A' }, { id: 21, text: ' dog' }] })
  render(<CompareView a={makeRunRecord(1)} b={b} />)
  expect(screen.getByTestId('cmp-badge')).toHaveTextContent('different prompts')
})

test('clicking a ruler tick opens paired distributions with the chosen token marked', () => {
  render(<CompareView a={makeRunRecord(1)} b={divergentB()} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[1])
  const sides = screen.getAllByTestId('cmp-dist-side')
  expect(sides).toHaveLength(2)
  const chosen = screen.getAllByTestId('cmp-bar-row').filter((r) => r.dataset.chosen === 'true')
  expect(chosen).toHaveLength(2)
})

test('a cycle past one run\'s end shows the run-ended side', () => {
  render(<CompareView a={makeRunRecord(1)} b={makeRunRecord(2, { cycles: 1 })} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[1])
  // B's side renders "run ended" in both the distributions and attention panels
  const ended = screen.getAllByTestId('cmp-ended')
  expect(ended.length).toBeGreaterThanOrEqual(1)
  expect(ended[0]).toHaveTextContent('run ended at cycle 0')
})

test('paired attention: shared heads render two heatmaps; a missing head falls back', () => {
  const a = makeRunRecord(1)
  const bBase = makeRunRecord(2)
  // strip L0·H3 from run B's attention events → that chip has no B-side matrix
  const b: RunRecord = { ...bBase, events: bBase.events.map((e) =>
    e.type === 'attention' ? { ...e, heads: e.heads.filter((h) => !(h.layer === 0 && h.head === 3)) } : e) }
  render(<CompareView a={a} b={b} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[0])
  const chips = screen.getAllByTestId('cmp-head-chip')
  expect(chips.map((c) => c.textContent)).toEqual(['attention-sink L0·H3', 'previous-token L2·H1'])
  // default selection = first chip (L0·H3): A side full heatmap, B side sim fallback note
  expect(screen.getAllByTestId('attention-heatmap')).toHaveLength(1)
  expect(screen.getByTestId('cmp-fallback')).toHaveTextContent("not among this run's detected heads")
  // select the shared head: both sides render full heatmaps
  fireEvent.click(chips[1])
  expect(screen.getAllByTestId('attention-heatmap')).toHaveLength(2)
})

test('a run with a grid uses the run-level thumbnail as fallback', () => {
  const a = makeRunRecord(1)
  const bBase = makeRunRecord(2, { mode: 'real' })
  const grid = makeGridEvent(2, 4)   // covers layer 0, head 3
  const b: RunRecord = { ...bBase, events: [
    ...bBase.events.slice(0, -1).map((e) =>
      e.type === 'attention' ? { ...e, heads: e.heads.filter((h) => !(h.layer === 0 && h.head === 3)) } : e),
    grid, bBase.events.at(-1)!,
  ] }
  render(<CompareView a={a} b={b} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[0])
  expect(screen.getByTestId('cmp-fallback')).toHaveTextContent('run-level thumbnail')
})

test('clicking a generated stream token selects its cycle', () => {
  render(<CompareView a={makeRunRecord(1)} b={makeRunRecord(2)} />)
  fireEvent.click(screen.getAllByTestId('cmp-token')[1])   // run A, cycle 1
  const selectedTokens = screen.getAllByTestId('cmp-token').filter((t) => t.dataset.selected === 'true')
  expect(selectedTokens).toHaveLength(2)                   // one per stream
  expect(screen.getAllByTestId('cmp-tick')[1].dataset.selected).toBe('true')
  expect(screen.getAllByTestId('cmp-dist-side')).toHaveLength(2)
})

test('ruler selection highlights the matching word chips in both streams', () => {
  render(<CompareView a={makeRunRecord(1)} b={makeRunRecord(2)} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[0])
  const selected = screen.getAllByTestId('cmp-token').filter((t) => t.dataset.selected === 'true')
  expect(selected.map((t) => t.textContent)).toEqual([' sat', ' sat'])
})

test('panel sides carry the run identity labels', () => {
  render(<CompareView a={makeRunRecord(3)} b={makeRunRecord(1)} />)
  fireEvent.click(screen.getAllByTestId('cmp-tick')[0])
  const sides = screen.getAllByTestId('cmp-dist-side')
  expect(sides[0]).toHaveTextContent('A #3')
  expect(sides[1]).toHaveTextContent('B #1')
})

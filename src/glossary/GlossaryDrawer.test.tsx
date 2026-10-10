import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { sampleRun } from '../test/glossaryFixture'
import { GlossaryDrawer } from './GlossaryDrawer'
import { getEntry } from './registry'
import { resetGlossaryStore, useGlossaryStore } from './store'

const scrollIntoView = vi.fn()
beforeEach(() => { Element.prototype.scrollIntoView = scrollIntoView })
afterEach(() => { cleanup(); resetGlossaryStore(); scrollIntoView.mockReset(); vi.useRealTimers() })
const open = (fn: () => void) => act(fn)

test('closed: renders nothing', () => {
  render(<GlossaryDrawer run={null} />)
  expect(screen.queryByTestId('glossary-drawer')).toBeNull()
})

test('entry view: title, stage chip, body, related links; heading gets focus', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('edit'))
  const drawer = screen.getByTestId('glossary-drawer')
  expect(drawer).toHaveAttribute('aria-label', 'Glossary')
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Edit')
  expect(drawer).toHaveTextContent('Layers')                      // stage chip
  expect(screen.getByTestId('gl-edit-figure')).toBeInTheDocument()
  expect(document.activeElement).toBe(screen.getByTestId('gl-heading'))
  expect(screen.getAllByTestId('gl-related').map((b) => b.textContent)).toEqual(['Residual stream', 'Attention', 'MLP'])
  expect(screen.queryByTestId('gl-in-run')).toBeNull()             // no run loaded
  expect(screen.queryByTestId('gl-back')).toBeNull()               // nothing to go back to
})

test('related link and Back navigate', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('edit'))
  fireEvent.click(screen.getAllByTestId('gl-related')[2])
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('MLP')
  fireEvent.click(screen.getByTestId('gl-back'))
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Edit')
})

test('In this run block shows the run’s numbers', () => {
  render(<GlossaryDrawer run={sampleRun()} />)
  open(() => useGlossaryStore.getState().openEntry('softmax'))
  const block = screen.getByTestId('gl-in-run')
  expect(block).toHaveTextContent('In this run')
  expect(block).toHaveTextContent('T = 0.8, 1 point ahead is 3.5×')
  expect(block).toHaveTextContent('␣sat 70%')
})

test('run becomes null: the block disappears, the entry stays', () => {
  const { rerender } = render(<GlossaryDrawer run={sampleRun()} />)
  open(() => useGlossaryStore.getState().openEntry('softmax'))
  rerender(<GlossaryDrawer run={null} />)
  expect(screen.queryByTestId('gl-in-run')).toBeNull()
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Softmax')
})

test('re-render with a new run keeps focus where it is', () => {
  const { rerender } = render(<><input data-testid="elsewhere" /><GlossaryDrawer run={sampleRun()} /></>)
  open(() => useGlossaryStore.getState().openEntry('softmax'))
  screen.getByTestId('elsewhere').focus()
  scrollIntoView.mockReset()
  rerender(<><input data-testid="elsewhere" /><GlossaryDrawer run={sampleRun({ cycle: 3 })} /></>)
  expect(document.activeElement).toBe(screen.getByTestId('elsewhere'))
  expect(scrollIntoView).not.toHaveBeenCalled()
})

test('a throwing inRun is skipped; the entry still renders', () => {
  const entry = getEntry('token')!
  const original = entry.inRun
  entry.inRun = () => { throw new Error('boom') }
  const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
  try {
    render(<GlossaryDrawer run={sampleRun()} />)
    open(() => useGlossaryStore.getState().openEntry('token'))
    expect(screen.getByTestId('gl-heading')).toHaveTextContent('Token')
    expect(screen.queryByTestId('gl-in-run')).toBeNull()
  } finally {
    entry.inRun = original
    spy.mockRestore()
  }
})

test('section link scrolls to the section and highlights it briefly', () => {
  vi.useFakeTimers()
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('softmax', 'temperature'))
  const section = screen.getByTestId('glossary-drawer').querySelector('[data-gl-section="temperature"]')!
  expect(scrollIntoView).toHaveBeenCalledTimes(1)
  expect(scrollIntoView.mock.contexts[0]).toBe(section)
  expect(section).toHaveClass('gl-flash')
  act(() => { vi.advanceTimersByTime(1600) })
  expect(section).not.toHaveClass('gl-flash')
})

test('unknown section: opens at the top without error', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('softmax', 'no-such-section'))
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Softmax')
  expect(scrollIntoView).not.toHaveBeenCalled()
})

test('Escape and × close the drawer', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('edit'))
  fireEvent.keyDown(document, { key: 'Escape' })
  expect(screen.queryByTestId('glossary-drawer')).toBeNull()
  open(() => useGlossaryStore.getState().openEntry('edit'))
  fireEvent.click(screen.getByTestId('gl-close'))
  expect(screen.queryByTestId('glossary-drawer')).toBeNull()
})

test('index: how-to-read first, then groups in pipeline order; a row opens its entry', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openIndex())
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Glossary')
  const rows = screen.getAllByTestId('gl-index-row')
  expect(rows[0].dataset.entry).toBe('how-to-read')
  expect(screen.getAllByTestId('gl-index-group').map((g) => g.dataset.stage))
    .toEqual(['tokenize', 'embed', 'layers', 'logits', 'softmax', 'sample', 'general'])
  expect(rows[0]).toHaveTextContent('A short tour of the screen')
  fireEvent.click(rows.find((r) => r.dataset.entry === 'logit')!)
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Logit')
  fireEvent.click(screen.getByTestId('gl-back'))
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Glossary')
})

test('“All entries” link opens the index from an entry', () => {
  render(<GlossaryDrawer run={null} />)
  open(() => useGlossaryStore.getState().openEntry('edit'))
  fireEvent.click(screen.getByTestId('gl-all-entries'))
  expect(screen.getByTestId('gl-heading')).toHaveTextContent('Glossary')
})

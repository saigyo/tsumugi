import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { resetGlossaryStore, useGlossaryStore } from './store'
import { Term } from './Term'
import type { GlossaryId } from './types'

afterEach(() => { cleanup(); resetGlossaryStore() })

test('hover shows the summary, leaving hides it', () => {
  render(<p>each block adds an <Term id="edit">edit</Term></p>)
  const term = screen.getByTestId('gl-term')
  expect(screen.queryByTestId('gl-tip')).toBeNull()
  fireEvent.mouseEnter(term)
  expect(screen.getByTestId('gl-tip')).toHaveTextContent('The vector a block adds')
  expect(screen.getByTestId('gl-tip')).toHaveTextContent('Click to read more.')
  expect(term).toHaveAttribute('aria-describedby', screen.getByTestId('gl-tip').id)
  fireEvent.mouseLeave(term)
  expect(screen.queryByTestId('gl-tip')).toBeNull()
})

test('keyboard focus shows the summary too', () => {
  render(<Term id="edit">edit</Term>)
  fireEvent.focus(screen.getByTestId('gl-term'))
  expect(screen.getByTestId('gl-tip')).toBeInTheDocument()
  fireEvent.blur(screen.getByTestId('gl-term'))
  expect(screen.queryByTestId('gl-tip')).toBeNull()
})

test('click opens the entry at the section and remembers the button as opener', () => {
  render(<Term id="embedding" section="learned" variant="chip" testId="embed-callout">ⓘ learned, not designed</Term>)
  const chip = screen.getByTestId('embed-callout')
  expect(chip).toHaveClass('embed-callout')
  expect(chip.dataset.term).toBe('embedding')
  expect(chip.dataset.section).toBe('learned')
  fireEvent.click(chip)
  const s = useGlossaryStore.getState()
  expect(s.view).toEqual({ kind: 'entry', id: 'embedding', section: 'learned' })
  expect(s.opener).toBe(chip)
  expect(screen.queryByTestId('gl-tip')).toBeNull()   // the tip closes when the drawer opens
})

test('icon variant carries an accessible name', () => {
  render(<Term id="token" variant="icon" label="About this stage">ⓘ</Term>)
  expect(screen.getByRole('button', { name: 'About this stage' })).toBeInTheDocument()
})

test('an id without an entry renders plain text, not a button', () => {
  render(<Term id={'nope' as GlossaryId}>plain words</Term>)
  expect(screen.queryByRole('button')).toBeNull()
  expect(screen.getByText('plain words')).toBeInTheDocument()
})

test('the tooltip is placed in the viewport, clamped away from the right edge', () => {
  render(<Term id="edit">edit</Term>)
  const term = screen.getByTestId('gl-term')
  term.getBoundingClientRect = () => ({ left: 990, right: 1020, top: 100, bottom: 118, width: 30, height: 18, x: 990, y: 100, toJSON: () => ({}) })
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1024 })
  fireEvent.mouseEnter(term)
  const tip = screen.getByTestId('gl-tip')
  expect(parseFloat(tip.style.left)).toBeLessThanOrEqual(1024 - 296)
  expect(tip.style.top).toBe('122px')
})

test('focus returned by the closing drawer does not bring the tooltip back', () => {
  render(<Term id="edit">edit</Term>)
  const term = screen.getByTestId('gl-term')
  fireEvent.click(term)                                   // opens the drawer, tooltip gone
  act(() => useGlossaryStore.getState().close('pointer'))  // focus comes back to the term
  expect(document.activeElement).toBe(term)
  expect(screen.queryByTestId('gl-tip')).toBeNull()
  act(() => { term.blur() })
  fireEvent.focus(term)                                    // a real later focus shows it again
  expect(screen.getByTestId('gl-tip')).toBeInTheDocument()
})

test('Escape hides the tooltip', () => {
  render(<Term id="edit">edit</Term>)
  fireEvent.mouseEnter(screen.getByTestId('gl-term'))
  fireEvent.keyDown(document, { key: 'Escape' })
  expect(screen.queryByTestId('gl-tip')).toBeNull()
})

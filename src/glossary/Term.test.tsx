import { cleanup, fireEvent, render, screen } from '@testing-library/react'
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

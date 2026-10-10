import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { resetGlossaryStore, useGlossaryStore } from '../glossary/store'
import { ResidualStreamDiagram } from './ResidualStreamDiagram'

afterEach(() => { cleanup(); resetGlossaryStore() })

test('the attention and MLP boxes open their entries by click and by keyboard', () => {
  render(<ResidualStreamDiagram seqLen={4} dims={576} layers={30} />)
  const attn = screen.getByTestId('rs-term-attention')
  expect(attn.querySelector('title')).toBeNull()
  fireEvent.click(attn)
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'attention' })
  resetGlossaryStore()
  fireEvent.keyDown(screen.getByTestId('rs-term-mlp'), { key: 'Enter' })
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'mlp' })
})

test('hover or focus on a box shows the entry summary', () => {
  render(<ResidualStreamDiagram seqLen={4} dims={576} layers={30} />)
  fireEvent.mouseEnter(screen.getByTestId('rs-term-mlp'))
  expect(screen.getByTestId('gl-tip')).toHaveTextContent('widen the vector')
  fireEvent.mouseLeave(screen.getByTestId('rs-term-mlp'))
  expect(screen.queryByTestId('gl-tip')).toBeNull()
  fireEvent.focus(screen.getByTestId('rs-term-attention'))
  expect(screen.getByTestId('gl-tip')).toHaveTextContent('gathers information')
})

test('the caption links edits and layers', () => {
  render(<ResidualStreamDiagram seqLen={4} dims={576} layers={30} />)
  const terms = screen.getAllByTestId('gl-term').map((t) => t.dataset.term)
  expect(terms).toEqual(['layer', 'edit'])
})

test('the boxes stay reachable for screen readers (the svg is a group, not an image)', () => {
  render(<ResidualStreamDiagram seqLen={4} dims={576} layers={30} />)
  const svg = screen.getByTestId('residual-diagram').querySelector('svg')!
  expect(svg.getAttribute('role')).toBe('group')
  expect(screen.getByRole('button', { name: /^MLP/ })).toBe(screen.getByTestId('rs-term-mlp'))
})

test('focus returned to a box by the closing drawer does not show its tooltip', () => {
  render(<ResidualStreamDiagram seqLen={4} dims={576} layers={30} />)
  const mlpBox = screen.getByTestId('rs-term-mlp')
  fireEvent.click(mlpBox)
  act(() => useGlossaryStore.getState().close('pointer'))
  expect(screen.queryByTestId('gl-tip')).toBeNull()
})

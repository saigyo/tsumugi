import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { makeGridEvent } from '../test/fixtures'
import { AttentionGridExplorer } from './AttentionGridExplorer'

// Mock getContext to prevent jsdom stub warnings in test output
vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null as never)

afterEach(() => cleanup())

test('layer sort renders one row per layer with an aggregate and all cells', () => {
  render(<AttentionGridExplorer grid={makeGridEvent(2, 2)} onPin={() => {}} />)
  expect(screen.getAllByTestId('grid-row')).toHaveLength(2)
  expect(screen.getAllByTestId('grid-aggregate')).toHaveLength(2)
  expect(screen.getAllByTestId('grid-cell')).toHaveLength(4)
  expect(screen.getAllByTestId('grid-cell')[0]).toHaveTextContent('L0·H0')
  expect(screen.getByTestId('grid-explorer')).toHaveTextContent('attention accumulated over the whole run')
})

test('sorting by distinctive reorders cells and drops aggregates', () => {
  render(<AttentionGridExplorer grid={makeGridEvent(2, 2)} onPin={() => {}} />)
  fireEvent.change(screen.getByTestId('grid-sort'), { target: { value: 'distinctive' } })
  // makeGridEvent(2,2) distinctive scores: L0·H1=0.8 > L1·H0=0.4 > L0·H0=0.2 > L1·H1=0
  expect(screen.getAllByTestId('grid-cell')[0]).toHaveTextContent('L0·H1')
  expect(screen.queryAllByTestId('grid-aggregate')).toHaveLength(0)
})

test('clicking a cell pins it', () => {
  const onPin = vi.fn()
  render(<AttentionGridExplorer grid={makeGridEvent(2, 2)} onPin={onPin} />)
  fireEvent.click(screen.getAllByTestId('grid-cell')[3])
  expect(onPin).toHaveBeenCalledWith(1, 1)
})

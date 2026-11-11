import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { HeaderTools } from './HeaderTools'

afterEach(() => cleanup())

const base = { runCount: 2, canArm: true, selecting: false, onArmCompare: () => {}, onExitCompare: () => {}, onImportFile: () => {} }

test('import is always present, even with no runs', () => {
  render(<HeaderTools {...base} runCount={0} />)
  expect(screen.getByTestId('btn-import')).toBeInTheDocument()
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
})

test('compare needs two runs', () => {
  const { rerender } = render(<HeaderTools {...base} runCount={1} />)
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
  rerender(<HeaderTools {...base} runCount={2} />)
  expect(screen.getByTestId('btn-compare-arm')).toBeEnabled()
})

test('compare is disabled without an active run', () => {
  render(<HeaderTools {...base} canArm={false} />)
  expect(screen.getByTestId('btn-compare-arm')).toBeDisabled()
})

test('exit replaces arm while selecting', () => {
  const onExitCompare = vi.fn()
  render(<HeaderTools {...base} selecting onExitCompare={onExitCompare} />)
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
  expect(screen.getByTestId('btn-compare-exit')).toHaveTextContent('Exit compare')
  fireEvent.click(screen.getByTestId('btn-compare-exit'))
  expect(onExitCompare).toHaveBeenCalled()
})

test('choosing a file calls onImportFile', () => {
  const onImportFile = vi.fn()
  render(<HeaderTools {...base} onImportFile={onImportFile} />)
  const file = new File(['{}'], 'run.json', { type: 'application/json' })
  fireEvent.change(screen.getByTestId('import-input'), { target: { files: [file] } })
  expect(onImportFile).toHaveBeenCalledWith(file)
})

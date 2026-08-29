import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { makeRunRecord } from '../test/fixtures'
import { RunShelf, type RunShelfProps } from './RunShelf'

afterEach(() => cleanup())

const props = (over: Partial<RunShelfProps> = {}): RunShelfProps => ({
  records: [makeRunRecord(1), makeRunRecord(2, { temperature: 0.2 })],
  activeId: 'run-2', compare: null, armed: false, persistFailed: false, importError: null,
  onActivate: vi.fn(), onSelectCompareB: vi.fn(), onArmCompare: vi.fn(), onExitCompare: vi.fn(),
  onTogglePin: vi.fn(), onExport: vi.fn(), onRemove: vi.fn(), onImportFile: vi.fn(),
  ...over,
})

test('renders nothing for an empty archive', () => {
  const { container } = render(<RunShelf {...props({ records: [] })} />)
  expect(container.firstChild).toBeNull()
})

test('chips show seq labels and the active run is marked', () => {
  render(<RunShelf {...props()} />)
  const chips = screen.getAllByTestId('run-chip')
  expect(chips).toHaveLength(2)
  expect(chips[0]).toHaveTextContent('#1 · The cat · T=0.8')
  expect(chips[1]).toHaveTextContent('T=0.2')
  expect(chips[0].dataset.active).toBe('false')
  expect(chips[1].dataset.active).toBe('true')
})

test('main click activates when not armed', () => {
  const p = props()
  render(<RunShelf {...p} />)
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])
  expect(p.onActivate).toHaveBeenCalledWith('run-1')
})

test('armed: clicking a non-active chip selects run B; the active chip is ignored', () => {
  const p = props({ armed: true })
  render(<RunShelf {...p} />)
  fireEvent.click(screen.getAllByTestId('run-chip-main')[1])   // active run = A
  expect(p.onSelectCompareB).not.toHaveBeenCalled()
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])
  expect(p.onSelectCompareB).toHaveBeenCalledWith('run-1')
  expect(p.onActivate).not.toHaveBeenCalled()
})

test('compare roles are marked and the exit control shows', () => {
  render(<RunShelf {...props({ compare: { aId: 'run-2', bId: 'run-1' } })} />)
  const chips = screen.getAllByTestId('run-chip')
  expect(chips[1].dataset.role).toBe('a')
  expect(chips[0].dataset.role).toBe('b')
  expect(screen.getByTestId('btn-compare-exit')).toBeInTheDocument()
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
})

test('a single record hides the compare affordance', () => {
  render(<RunShelf {...props({ records: [makeRunRecord(1)], activeId: 'run-1' })} />)
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
})

test('pinned marker, pin/remove/export actions', () => {
  const p = props({ records: [makeRunRecord(1, { pinned: true })], activeId: 'run-1' })
  render(<RunShelf {...p} />)
  expect(screen.getByTestId('run-chip').dataset.pinned).toBe('true')
  // the single pin button is the toggle AND the state cue — no second pin in the label
  expect(screen.getByTestId('btn-chip-pin').dataset.pinned).toBe('true')
  expect(screen.getByTestId('btn-chip-pin')).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByTestId('run-chip-main').textContent).not.toContain('📌')
  fireEvent.click(screen.getByTestId('btn-chip-pin'))
  expect(p.onTogglePin).toHaveBeenCalledWith('run-1')
  fireEvent.click(screen.getByTestId('btn-chip-export'))
  expect(p.onExport).toHaveBeenCalledWith('run-1')
  fireEvent.click(screen.getByTestId('btn-chip-remove'))
  expect(p.onRemove).toHaveBeenCalledWith('run-1')
})

test('import file input forwards the file; notes render', () => {
  const p = props({ persistFailed: true, importError: 'bad file' })
  render(<RunShelf {...p} />)
  const file = new File(['{}'], 'run.json', { type: 'application/json' })
  fireEvent.change(screen.getByTestId('import-input'), { target: { files: [file] } })
  expect(p.onImportFile).toHaveBeenCalledWith(file)
  expect(screen.getByTestId('shelf-note')).toHaveTextContent('not persisted')
  expect(screen.getByTestId('import-error')).toHaveTextContent('bad file')
})

test('exit-compare stays reachable when only one record remains while comparing', () => {
  const p = props({ records: [makeRunRecord(1)], compare: { aId: 'run-1', bId: 'gone' } })
  render(<RunShelf {...p} />)
  expect(screen.getByTestId('btn-compare-exit')).toBeInTheDocument()
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
  fireEvent.click(screen.getByTestId('btn-compare-exit'))
  expect(p.onExitCompare).toHaveBeenCalled()
})

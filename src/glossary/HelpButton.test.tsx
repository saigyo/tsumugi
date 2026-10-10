import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { HelpButton } from './HelpButton'
import { resetGlossaryStore, useGlossaryStore } from './store'

const KEY = 'tsumugi.glossaryHintDismissed'
// Node 26 defines its own global localStorage (undefined without
// --localstorage-file), which hides jsdom's; tests bring an in-memory one
function memoryStorage() {
  const data = new Map<string, string>()
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => { data.set(k, String(v)) },
    removeItem: (k: string) => { data.delete(k) },
    clear: () => data.clear(),
  }
}
beforeEach(() => { vi.stubGlobal('localStorage', memoryStorage()) })
afterEach(() => { cleanup(); resetGlossaryStore(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

test('the ? button opens the index', () => {
  render(<HelpButton />)
  fireEvent.click(screen.getByTestId('btn-help'))
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'index' })
  expect(useGlossaryStore.getState().opener).toBe(screen.getByTestId('btn-help'))
})

test('first visit shows the hint; following it opens How to read and dismisses it for good', () => {
  render(<HelpButton />)
  expect(screen.getByTestId('glossary-hint')).toHaveTextContent('New here? Start with How to read Tsumugi')
  fireEvent.click(screen.getByTestId('btn-hint-open'))
  expect(useGlossaryStore.getState().view).toEqual({ kind: 'entry', id: 'how-to-read' })
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
  expect(localStorage.getItem(KEY)).toBe('1')
})

test('× dismisses the hint and it stays gone on the next visit', () => {
  const first = render(<HelpButton />)
  fireEvent.click(screen.getByTestId('btn-hint-dismiss'))
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
  first.unmount()
  render(<HelpButton />)
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
})

test('opening any entry elsewhere dismisses the hint', () => {
  render(<HelpButton />)
  act(() => useGlossaryStore.getState().openEntry('edit'))
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
  expect(localStorage.getItem(KEY)).toBe('1')
})

test('storage that throws: the hint still shows and dismissing does not error', () => {
  const blocked = () => { throw new Error('blocked') }
  vi.stubGlobal('localStorage', { getItem: blocked, setItem: blocked, removeItem: blocked, clear: blocked })
  render(<HelpButton />)
  expect(screen.getByTestId('glossary-hint')).toBeInTheDocument()
  expect(() => fireEvent.click(screen.getByTestId('btn-hint-dismiss'))).not.toThrow()
  expect(screen.queryByTestId('glossary-hint')).toBeNull()
})

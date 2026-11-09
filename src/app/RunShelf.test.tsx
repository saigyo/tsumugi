import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { makeRunRecord } from '../test/fixtures'
import type { RunEndReason } from '../trace/types'
import type { RunRecord } from './runsStore'
import { RunShelf, type RunShelfProps } from './RunShelf'

afterEach(() => cleanup())

const props = (over: Partial<RunShelfProps> = {}): RunShelfProps => ({
  records: [makeRunRecord(1), makeRunRecord(2, { temperature: 0.2 })],
  activeId: 'run-2', compare: null, armed: false, persistFailed: false, importError: null,
  onActivate: vi.fn(), onSelectCompareB: vi.fn(), 
  onTogglePin: vi.fn(), onExport: vi.fn(), onRemove: vi.fn(),
  ...over,
})

test('renders nothing for an empty archive without notes', () => {
  const { container } = render(<RunShelf {...props({ records: [] })} />)
  expect(container.firstChild).toBeNull()
})

test('an empty archive still shows the notes line', () => {
  render(<RunShelf {...props({ records: [], importError: 'bad file' })} />)
  expect(screen.getByTestId('import-error')).toHaveTextContent('bad file')
})

test('the shelf no longer holds import or compare buttons', () => {
  render(<RunShelf {...props()} />)
  expect(screen.queryByTestId('btn-import')).toBeNull()
  expect(screen.queryByTestId('btn-compare-arm')).toBeNull()
})

test('chips show seq labels and the active run is marked', () => {
  render(<RunShelf {...props()} />)
  const chips = screen.getAllByTestId('run-chip')
  expect(chips).toHaveLength(2)
  expect(chips[0]).toHaveTextContent('#2 · The cat')
  expect(chips[1]).toHaveTextContent('#1 · The cat')
  expect(chips[0].dataset.active).toBe('true')
  expect(chips[1].dataset.active).toBe('false')
})

test('main click activates when not armed', () => {
  const p = props()
  render(<RunShelf {...p} />)
  fireEvent.click(screen.getAllByTestId('run-chip-main')[1])
  expect(p.onActivate).toHaveBeenCalledWith('run-1')
})

test('armed: clicking a non-active chip selects run B; the active chip is ignored', () => {
  const p = props({ armed: true })
  render(<RunShelf {...p} />)
  fireEvent.click(screen.getAllByTestId('run-chip-main')[0])   // active run = A
  expect(p.onSelectCompareB).not.toHaveBeenCalled()
  fireEvent.click(screen.getAllByTestId('run-chip-main')[1])
  expect(p.onSelectCompareB).toHaveBeenCalledWith('run-1')
  expect(p.onActivate).not.toHaveBeenCalled()
})

test('compare roles are marked', () => {
  render(<RunShelf {...props({ compare: { aId: 'run-2', bId: 'run-1' } })} />)
  const chips = screen.getAllByTestId('run-chip')
  expect(chips[0].dataset.role).toBe('a')
  expect(chips[1].dataset.role).toBe('b')
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

test('notes render', () => {
  const p = props({ persistFailed: true, importError: 'bad file' })
  render(<RunShelf {...p} />)
  expect(screen.getByTestId('shelf-note')).toHaveTextContent('not persisted')
  expect(screen.getByTestId('import-error')).toHaveTextContent('bad file')
})

test('a chat run’s chip leaves out Chat and its title names the system prompt', () => {
  const chat = makeRunRecord(1, { mode: 'real', prompt: 'What is the capital of France?' })
  chat.meta.format = { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' }
  const none = makeRunRecord(2, { mode: 'real', prompt: 'Hi there' })
  none.meta.format = { kind: 'chat', system: null }
  const plain = makeRunRecord(3, { prompt: 'The cat sat' })
  render(<RunShelf {...props({ records: [chat, none, plain] })} />)
  const mains = screen.getAllByTestId('run-chip-main')   // newest first
  expect(mains[2]).toHaveTextContent('#1 · What is the')
  expect(mains[2]).not.toHaveTextContent('Chat')
  expect(mains[2]).toHaveAttribute('title', expect.stringContaining('system: default'))
  expect(mains[1]).toHaveAttribute('title', expect.stringContaining('system: none'))
  expect(mains[0]).toHaveTextContent('#3 · The cat sat')
  expect(mains[0].getAttribute('title')).not.toContain('system:')
})

test('a custom system prompt is named in the title; one longer than 60 characters is cut with an ellipsis', () => {
  const short = makeRunRecord(1, { mode: 'real', prompt: 'Hi' })
  short.meta.format = { kind: 'chat', system: 'You always answer in French.' }
  const long = makeRunRecord(2, { mode: 'real', prompt: 'Hi' })
  long.meta.format = { kind: 'chat', system: 'x'.repeat(60) + 'TAIL' }
  const exact = makeRunRecord(3, { mode: 'real', prompt: 'Hi' })
  exact.meta.format = { kind: 'chat', system: 'y'.repeat(60) }
  render(<RunShelf {...props({ records: [short, long, exact] })} />)
  const titles = screen.getAllByTestId('run-chip-main').map((m) => m.getAttribute('title'))
  expect(titles[2]).toContain(' · system: You always answer in French. · ended:')
  expect(titles[1]).toContain(` · system: ${'x'.repeat(60)}… · ended:`)
  expect(titles[0]).toContain(` · system: ${'y'.repeat(60)} · ended:`)
})

test('a knockout run chip label ends with the zeroed layer', () => {
  const r = makeRunRecord(3)
  r.meta.knockout = { layer: 15, cycle: 0, sourceId: 'run-1' }
  render(<RunShelf {...props({ records: [r] })} />)
  expect(screen.getByTestId('run-chip-main')).toHaveTextContent(/· MLP 15 off$/)
})

// the fixture has no seed option: a real, seeded, ended run is built by patching run-start
const seeded = (seq: number, patch: Record<string, unknown> = {}, reason: RunEndReason = 'max-tokens'): RunRecord => {
  const r = makeRunRecord(seq, { mode: 'real', reason })
  return { ...r, events: r.events.map((e) => (e.type === 'run-start' ? { ...e, seed: 7, ...patch } : e)) }
}

const stubClipboard = (writeText: (t: string) => Promise<void>) =>
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })

test("a shareable run's chip has a share button between pin and download", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined)
  stubClipboard(writeText)
  render(<RunShelf {...props({ records: [seeded(1)], activeId: 'run-1' })} />)
  const share = screen.getByTestId('btn-chip-share')
  const order = [...screen.getByTestId('run-chip').querySelectorAll('button')] as HTMLElement[]
  expect(order.indexOf(share)).toBe(order.indexOf(screen.getByTestId('btn-chip-pin')) + 1)
  expect(order.indexOf(screen.getByTestId('btn-chip-export'))).toBe(order.indexOf(share) + 1)
  await act(async () => { fireEvent.click(share) })
  expect(writeText).toHaveBeenCalledTimes(1)
  expect(writeText.mock.calls[0][0]).toContain('seed=')
  expect(screen.getByText('Link copied')).toBeInTheDocument()
})

test('no share button for sim, seedless, knockout or unfinished runs', () => {
  const sim = makeRunRecord(1)
  const seedless = seeded(2, { seed: undefined })
  const knockout = seeded(3, { knockout: { layer: 1 } })
  const aborted = seeded(4, {}, 'aborted')
  render(<RunShelf {...props({ records: [sim, seedless, knockout, aborted], activeId: null })} />)
  expect(screen.queryByTestId('btn-chip-share')).toBeNull()
})

test('a failed copy shows the link in a readonly field', async () => {
  stubClipboard(vi.fn().mockRejectedValue(new Error('denied')))
  render(<RunShelf {...props({ records: [seeded(1)], activeId: 'run-1' })} />)
  await act(async () => { fireEvent.click(screen.getByTestId('btn-chip-share')) })
  const field = screen.getByTestId('share-link-text') as HTMLInputElement
  expect(field.readOnly).toBe(true)
  expect(field.value).toContain('seed=')
})

test('"Link copied" goes away when the run is removed within the 2 s window, and after 2 s', async () => {
  vi.useFakeTimers()
  try {
    stubClipboard(vi.fn().mockResolvedValue(undefined))
    const a = seeded(1), b = seeded(2)
    const { rerender } = render(<RunShelf {...props({ records: [a, b], activeId: 'run-1' })} />)
    await act(async () => { fireEvent.click(screen.getAllByTestId('btn-chip-share')[0]) })
    expect(screen.getByText('Link copied')).toBeInTheDocument()
    rerender(<RunShelf {...props({ records: [a], activeId: 'run-1' })} />)
    expect(screen.queryByText('Link copied')).toBeNull()
    await act(async () => { fireEvent.click(screen.getByTestId('btn-chip-share')) })
    expect(screen.getByText('Link copied')).toBeInTheDocument()
    await act(async () => { vi.advanceTimersByTime(2100) })
    expect(screen.queryByText('Link copied')).toBeNull()
  } finally { vi.useRealTimers() }
})

const many = (n: number) => Array.from({ length: n }, (_, i) => makeRunRecord(i + 1))

test('eight runs: five chips newest first and a "+3 more" button; five or fewer have none', () => {
  const { rerender } = render(<RunShelf {...props({ records: many(8), activeId: 'run-8' })} />)
  expect(screen.getAllByTestId('run-chip').map((c) => c.textContent)).toHaveLength(5)
  expect(screen.getAllByTestId('run-chip-main')[0]).toHaveTextContent('#8')
  expect(screen.getAllByTestId('run-chip-main')[4]).toHaveTextContent('#4')
  expect(screen.getByTestId('run-more')).toHaveTextContent('+3 more ▾')
  rerender(<RunShelf {...props({ records: many(5), activeId: 'run-5' })} />)
  expect(screen.queryByTestId('run-more')).toBeNull()
})

test('an older active run stays in the row; without it the row goes back to five', () => {
  const { rerender } = render(<RunShelf {...props({ records: many(8), activeId: 'run-2' })} />)
  expect(screen.getAllByTestId('run-chip')).toHaveLength(6)
  expect(screen.getByTestId('run-more')).toHaveTextContent('+2 more ▾')
  rerender(<RunShelf {...props({ records: many(8), activeId: null })} />)
  expect(screen.getAllByTestId('run-chip')).toHaveLength(5)
  expect(screen.getByTestId('run-more')).toHaveTextContent('+3 more ▾')
})

test('the list shows every run; choosing one activates it and closes the list', () => {
  const p = props({ records: many(8), activeId: 'run-8' })
  render(<RunShelf {...p} />)
  expect(screen.queryByTestId('run-list')).toBeNull()
  fireEvent.click(screen.getByTestId('run-more'))
  expect(screen.getByTestId('run-more')).toHaveAttribute('aria-expanded', 'true')
  expect(screen.getAllByTestId('run-list-row')).toHaveLength(8)
  fireEvent.click(screen.getAllByTestId('run-list-main')[7])
  expect(p.onActivate).toHaveBeenCalledWith('run-1')
  expect(screen.queryByTestId('run-list')).toBeNull()
})

test('Escape closes the list and focuses the button; dropping to five runs removes it', () => {
  const { rerender } = render(<RunShelf {...props({ records: many(8), activeId: 'run-8' })} />)
  fireEvent.click(screen.getByTestId('run-more'))
  fireEvent.keyDown(document, { key: 'Escape' })
  expect(screen.queryByTestId('run-list')).toBeNull()
  expect(screen.getByTestId('run-more')).toHaveFocus()
  fireEvent.click(screen.getByTestId('run-more'))
  rerender(<RunShelf {...props({ records: many(5), activeId: 'run-5' })} />)
  expect(screen.queryByTestId('run-list')).toBeNull()
})

test('an outside click closes the list', () => {
  render(<RunShelf {...props({ records: many(8), activeId: 'run-8' })} />)
  fireEvent.click(screen.getByTestId('run-more'))
  fireEvent.mouseDown(document.body)
  expect(screen.queryByTestId('run-list')).toBeNull()
})

test('armed: a list row picks run B and closes the list', () => {
  const p = props({ records: many(8), activeId: 'run-8', armed: true })
  render(<RunShelf {...p} />)
  fireEvent.click(screen.getByTestId('run-more'))
  fireEvent.click(screen.getAllByTestId('run-list-main')[7])
  expect(p.onSelectCompareB).toHaveBeenCalledWith('run-1')
  expect(screen.queryByTestId('run-list')).toBeNull()
})

test('chip and list text for a chat run at T=0.8', () => {
  const chat = makeRunRecord(16, { mode: 'real', prompt: 'What is the capital of France?' })
  chat.meta.format = { kind: 'chat', system: null }
  render(<RunShelf {...props({ records: [...many(6).map((r, i) => ({ ...r, meta: { ...r.meta, seq: i + 1 } })), chat], activeId: 'run-16' })} />)
  const main = screen.getAllByTestId('run-chip-main')[0]
  expect(main.textContent).toMatch(/#16 · What is the$/)
  expect(main.getAttribute('title')).toContain('T=0.8')
  fireEvent.click(screen.getByTestId('run-more'))
  expect(screen.getAllByTestId('run-list-main')[0].textContent).toMatch(/#16 · Chat · What is the · T=0\.8$/)
})

test('list rows have pin, export and remove buttons wired to the same handlers', () => {
  const p = props({ records: many(8), activeId: 'run-8' })
  render(<RunShelf {...p} />)
  fireEvent.click(screen.getByTestId('run-more'))
  const row = screen.getAllByTestId('run-list-row')[7]
  fireEvent.click(within(row).getByTestId('btn-chip-pin'))
  fireEvent.click(within(row).getByTestId('btn-chip-export'))
  fireEvent.click(within(row).getByTestId('btn-chip-remove'))
  expect(p.onTogglePin).toHaveBeenCalledWith('run-1')
  expect(p.onExport).toHaveBeenCalledWith('run-1')
  expect(p.onRemove).toHaveBeenCalledWith('run-1')
})

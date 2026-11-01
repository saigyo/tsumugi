import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { buildFixtureTrace } from '../test/fixtures'
import { cycleVotes, type VotedCandidate } from './selectors'
import { LayersNav } from './LayersNav'
import { LAYER_SECTIONS } from './layersSections'
import { useMlpLayer } from './mlpLayerSelection'

const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true })
const candidates = (cycleVotes(events, events.length - 1) as { candidates: VotedCandidate[] }).candidates
const run = events[0]

function page(tops: number[]) {
  return render(<>
    <LayersNav candidates={candidates} run={run} showStepper />
    {LAYER_SECTIONS.map((s, i) => {
      return <div key={s.id} id={s.id} ref={(el) => { if (el) el.getBoundingClientRect = () => ({ top: tops[i] } as DOMRect) }} />
    })}
  </>)
}

beforeEach(() => { Element.prototype.scrollIntoView = vi.fn() })
afterEach(() => { cleanup(); vi.useRealTimers() })

test('seven links in pipeline order, in a labelled nav landmark', () => {
  page([0, 500, 1000, 1500, 2000, 2500, 3000])
  expect(screen.getByRole('navigation', { name: 'Layers sections' })).toBeInTheDocument()
  expect(screen.getAllByTestId('nav-link').map((l) => l.textContent)).toEqual(['Overview', 'Attention', 'Edit sizes', 'Votes', 'Knockouts', 'Inside one MLP', 'Step by step'])
})

test('a click scrolls its section into view', () => {
  page([0, 500, 1000, 1500, 2000, 2500, 3000])
  fireEvent.click(screen.getAllByTestId('nav-link')[6])
  expect(Element.prototype.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' })
})

test('with reduced motion requested, a click jumps without smooth scrolling', () => {
  const original = window.matchMedia
  window.matchMedia = vi.fn((q: string) => ({ matches: q === '(prefers-reduced-motion: reduce)', media: q } as MediaQueryList))
  try {
    page([0, 500, 1000, 1500, 2000, 2500, 3000])
    fireEvent.click(screen.getAllByTestId('nav-link')[2])
    expect(Element.prototype.scrollIntoView).toHaveBeenCalledWith({ behavior: 'auto', block: 'start' })
  } finally { window.matchMedia = original }
})

test('the section whose top has passed under the bar is current', () => {
  vi.useFakeTimers()
  page([-1200, -700, -200, -10, 300, 600, 1100])
  act(() => { window.dispatchEvent(new Event('scroll')); vi.advanceTimersByTime(50) })
  const current = screen.getAllByTestId('nav-link').filter((l) => l.getAttribute('aria-current') === 'true')
  expect(current.map((l) => l.textContent)).toEqual(['Votes'])
})

test('the stepper shows and moves the shared MLP layer', () => {
  useMlpLayer.getState().select(run, 1)
  page([0, 1, 2, 3, 4, 5, 6])
  expect(screen.getByTestId('nav-layer')).toHaveTextContent('MLP layer 1')
  fireEvent.click(screen.getByTestId('nav-next'))
  expect(screen.getByTestId('nav-layer')).toHaveTextContent('MLP layer 2')
})

test('without MLP data the stepper is hidden and the links stay', () => {
  render(<LayersNav run={run} showStepper={false} />)
  expect(screen.queryByTestId('nav-layer')).toBeNull()
  expect(screen.getAllByTestId('nav-link')).toHaveLength(7)
})

test('a section that a jump left a few pixels lower than its scroll margin still counts as reached', () => {
  vi.useFakeTimers()
  // jsdom's bar ends at 0; the scroll margin puts a jumped-to section 6 px below it,
  // and fonts or rounding on another machine can leave it a little lower
  page([-3500, -3000, -2500, -2000, -1500, -1000, 20])
  act(() => { window.dispatchEvent(new Event('scroll')); vi.advanceTimersByTime(50) })
  const current = () => screen.getAllByTestId('nav-link').filter((l) => l.getAttribute('aria-current') === 'true').map((l) => l.textContent)
  expect(current()).toEqual(['Step by step'])
  cleanup()
  page([-3500, -3000, -2500, -2000, -1500, -1000, 120])
  act(() => { window.dispatchEvent(new Event('scroll')); vi.advanceTimersByTime(50) })
  expect(current()).toEqual(['Inside one MLP'])
})

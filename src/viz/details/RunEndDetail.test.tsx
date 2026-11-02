import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import type { TraceEvent } from '../../trace/types'
import { buildShareLink, shareableRun } from '../../app/shareLink'
import { resetRunRequest, useRunRequest } from '../runRequest'
import { RunEndDetail } from './RunEndDetail'
import { ShareLinkButton } from '../ShareLinkButton'

type Start = Extract<TraceEvent, { type: 'run-start' }>
type End = Extract<TraceEvent, { type: 'run-end' }>

const start = (over: Partial<Start> = {}): Start => ({
  type: 'run-start', prompt: 'Once upon', mode: 'real', modelId: 'm', seed: 123456789, revision: 'v7',
  params: { temperature: 0.8, topK: 10, maxNewTokens: 20 }, ...over,
})
const end = (over: Partial<End> = {}): End => ({ type: 'run-end', reason: 'max-tokens', ...over } as End)

const link = (events: TraceEvent[]) => buildShareLink(location.origin + location.pathname, shareableRun(events)!)

function view(s: Start, e: End, truncated = false) {
  const events: TraceEvent[] = [s, e]
  return render(<RunEndDetail runStart={s} runEnd={e} promptTokens={3} generatedTokens={20}
    shareable={shareableRun(events)} truncated={truncated} />)
}

const writeText = vi.fn()
beforeEach(() => {
  writeText.mockReset().mockResolvedValue(undefined)
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
  resetRunRequest()
  useRunRequest.setState({ model: { revision: 'v7' }, busy: false, continueRun: vi.fn() })
})
afterEach(() => { cleanup(); vi.useRealTimers() })

test('shows the seed in the settings line', () => {
  view(start(), end())
  expect(screen.getByTestId('detail-run-end').textContent).toContain('seed 123456789')
})

test('continue doubles the limit', () => {
  view(start(), end())
  fireEvent.click(screen.getByRole('button', { name: 'Run again with 40 tokens' }))
  expect(useRunRequest.getState().continueRun).toHaveBeenCalledWith(40)
})

test('no continue at 100 tokens', () => {
  view(start({ params: { temperature: 0.8, topK: 10, maxNewTokens: 100 } }), end())
  expect(screen.queryByRole('button', { name: /Run again/ })).toBeNull()
})
test('no continue after eos', () => {
  view(start(), end({ reason: 'eos' }))
  expect(screen.queryByRole('button', { name: /Run again/ })).toBeNull()
})
test('no continue for a truncated prompt', () => {
  view(start(), end(), true)
  expect(screen.queryByRole('button', { name: /Run again/ })).toBeNull()
})

test('continue is disabled for another model version', () => {
  useRunRequest.setState({ model: { revision: 'v8' } })
  view(start(), end())
  const b = screen.getByRole('button', { name: /Run again/ })
  expect(b).toBeDisabled()
  expect(b).toHaveAttribute('title', 'Load model version v7 to continue this run')
})
test('continue is disabled while the model is not loaded', () => {
  useRunRequest.setState({ model: null })
  view(start(), end())
  const b = screen.getByRole('button', { name: /Run again/ })
  expect(b).toBeDisabled()
  expect(b).toHaveAttribute('title', 'Load model version v7 to continue this run')
})
test('a stock run needs the stock model', () => {
  useRunRequest.setState({ model: { revision: 'v7' } })
  view(start({ revision: undefined }), end())
  expect(screen.getByRole('button', { name: /Run again/ })).toHaveAttribute('title', 'Load model version stock to continue this run')
})

test('copy link writes the share link', async () => {
  view(start(), end())
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy link' })) })
  expect(writeText).toHaveBeenCalledWith(link([start(), end()]))
  expect(screen.getByText('Link copied')).toBeInTheDocument()
})

test('Link copied clears after 2 seconds', async () => {
  vi.useFakeTimers()
  view(start(), end())
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy link' })) })
  expect(screen.getByText('Link copied')).toBeInTheDocument()
  act(() => { vi.advanceTimersByTime(2000) })
  expect(screen.queryByText('Link copied')).toBeNull()
})

test('copy failure shows the link in a text field', async () => {
  writeText.mockRejectedValue(new Error('denied'))
  view(start(), end())
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy link' })) })
  const input = screen.getByTestId('share-link-text') as HTMLInputElement
  expect(input).toHaveAttribute('readonly')
  expect(input.value).toBe(link([start(), end()]))
})

test.each([
  ['sim', start({ mode: 'sim' }), end()],
  ['seedless', start({ seed: undefined }), end()],
  ['knockout', start({ knockout: { sourceRunId: 'r', layer: 1, cycle: 0 } as never }), end()],
  ['aborted', start(), end({ reason: 'aborted' })],
  ['error', start(), end({ reason: 'error' })],
])('nothing extra for %s runs', (_n, s, e) => {
  view(s, e)
  expect(screen.getByTestId('detail-run-end').textContent).not.toContain('seed')
  expect(screen.queryByRole('button', { name: 'Copy link' })).toBeNull()
  expect(screen.queryByRole('button', { name: /Run again/ })).toBeNull()
})

test('buttons are disabled while busy', () => {
  useRunRequest.setState({ busy: true })
  view(start(), end())
  expect(screen.getByRole('button', { name: 'Copy link' })).toBeDisabled()
  expect(screen.getByRole('button', { name: /Run again/ })).toBeDisabled()
})

test('onFeedback receives the result instead of inline feedback', async () => {
  vi.useFakeTimers()
  const onFeedback = vi.fn()
  const s = shareableRun([start(), end()])!
  render(<ShareLinkButton settings={s} onFeedback={onFeedback} />)
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy link' })) })
  expect(onFeedback).toHaveBeenLastCalledWith({ kind: 'copied' })
  expect(screen.queryByText('Link copied')).toBeNull()
  act(() => { vi.advanceTimersByTime(2000) })
  expect(onFeedback).toHaveBeenLastCalledWith(null)
  writeText.mockRejectedValue(new Error('x'))
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Copy link' })) })
  expect(onFeedback).toHaveBeenLastCalledWith({ kind: 'failed', link: link([start(), end()]) })
  expect(screen.queryByTestId('share-link-text')).toBeNull()
})

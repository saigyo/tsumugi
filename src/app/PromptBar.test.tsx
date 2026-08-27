import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { PromptBar } from './PromptBar'

afterEach(() => cleanup())

const noop = () => {}

test('generate disabled for empty prompt', () => {
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={noop} busy={false} />)
  expect(screen.getByTestId('btn-generate')).toBeDisabled()
})

test('generate passes prompt and params', () => {
  const onGenerate = vi.fn()
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'The cat' } })
  fireEvent.change(screen.getByTestId('temp-input'), { target: { value: '0.5' } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenCalledWith('The cat', { temperature: 0.5, topK: 10, maxNewTokens: 20 })
})

test('mode toggle reports changes', () => {
  const onModeChange = vi.fn()
  render(<PromptBar mode="sim" onModeChange={onModeChange} onGenerate={noop} busy={false} />)
  fireEvent.click(screen.getByTestId('mode-toggle'))
  expect(onModeChange).toHaveBeenCalledWith('real')
})

test('out-of-range typed values are clamped; NaN keeps the previous value', () => {
  const onGenerate = vi.fn()
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'Hi' } })
  fireEvent.change(screen.getByTestId('topk-input'), { target: { value: '0' } })    // below min → 1
  fireEvent.change(screen.getByTestId('temp-input'), { target: { value: '9' } })    // above max → 2
  fireEvent.change(screen.getByTestId('maxtok-input'), { target: { value: '' } })   // NaN → keep 20
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenCalledWith('Hi', { temperature: 2, topK: 1, maxNewTokens: 20 })
})

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

test('example chips fill the prompt and start generation', () => {
  const onGenerate = vi.fn()
  const examples = [{ id: 'x', label: 'Example X', prompt: 'The cat sat', hint: 'watch the cat' }]
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={onGenerate} busy={false} examples={examples} />)
  const chip = screen.getByTestId('example-chip')
  expect(chip).toHaveTextContent('Example X')
  expect(chip).toHaveAttribute('title', 'watch the cat')
  fireEvent.click(chip)
  expect(screen.getByTestId('prompt-input')).toHaveValue('The cat sat')
  expect(onGenerate).toHaveBeenCalledWith('The cat sat', { temperature: 0.8, topK: 10, maxNewTokens: 20 })
})

test('status slot renders inside the config row', () => {
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={noop} busy={false}
    status={<span data-testid="fake-status">webgpu</span>} />)
  const slot = screen.getByTestId('model-status-slot')
  expect(slot).toContainElement(screen.getByTestId('fake-status'))
})

test('example chips are hidden while busy', () => {
  const examples = [{ id: 'x', label: 'Example X', prompt: 'p', hint: 'h' }]
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={noop} busy={true} examples={examples} />)
  expect(screen.getByTestId('example-chip')).toBeDisabled()
})

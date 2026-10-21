import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { PromptBar } from './PromptBar'
import { CHAT_EXAMPLE } from './chatExample'

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
  expect(onGenerate).toHaveBeenCalledWith('The cat', { temperature: 0.5, topK: 10, maxNewTokens: 20 }, { kind: 'text' })
})

test('the mode switch reports changes and shows the current mode', () => {
  const onModeChange = vi.fn()
  render(<PromptBar mode="sim" onModeChange={onModeChange} onGenerate={noop} busy={false} />)
  expect(screen.getByTestId('mode-sim')).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByTestId('mode-real')).toHaveTextContent('Real model · 250 MB')
  fireEvent.click(screen.getByTestId('mode-real'))
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
  expect(onGenerate).toHaveBeenCalledWith('Hi', { temperature: 2, topK: 1, maxNewTokens: 20 }, { kind: 'text' })
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
  expect(onGenerate).toHaveBeenCalledWith('The cat sat', { temperature: 0.8, topK: 10, maxNewTokens: 20 }, { kind: 'text' })
})

test('sampling inputs carry explanatory tooltips', () => {
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={noop} busy={false} />)
  expect(screen.getByTestId('temp-input').closest('label')).toHaveAttribute('title', expect.stringContaining('Temperature'))
  expect(screen.getByTestId('topk-input').closest('label')).toHaveAttribute('title', expect.stringContaining('Top-k'))
  expect(screen.getByTestId('maxtok-input').closest('label')).toHaveAttribute('title', expect.stringContaining('loop bound'))
})

test('example chips are hidden while busy', () => {
  const examples = [{ id: 'x', label: 'Example X', prompt: 'p', hint: 'h' }]
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={noop} busy={true} examples={examples} />)
  expect(screen.getByTestId('example-chip')).toBeDisabled()
})

test('the format switch and the system line exist only with the real model', () => {
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={noop} busy={false} />)
  expect(screen.queryByTestId('format-switch')).toBeNull()
  expect(screen.queryByTestId('system-input')).toBeNull()
  cleanup()
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={noop} busy={false} />)
  expect(screen.getByTestId('format-chat')).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByTestId('system-input')).toHaveValue('You are a helpful AI assistant named SmolLM, trained by Hugging Face')
  expect(screen.getByTestId('prompt-input')).toHaveAttribute('placeholder', 'Ask a question…')
  fireEvent.click(screen.getByTestId('format-text'))
  expect(screen.queryByTestId('system-input')).toBeNull()
  expect(screen.getByTestId('prompt-input')).toHaveAttribute('placeholder', 'Type a prompt…')
})

test('Chat sends the system prompt; empty means none; Reset restores the default', () => {
  const onGenerate = vi.fn()
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'Hi' } })
  expect(screen.queryByTestId('system-reset')).toBeNull()
  fireEvent.change(screen.getByTestId('system-input'), { target: { value: '  ' } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenLastCalledWith('Hi', expect.anything(), { kind: 'chat', system: null })
  fireEvent.click(screen.getByTestId('system-reset'))
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenLastCalledWith('Hi', expect.anything(),
    { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' })
})

test('Chat selected, then Simulated: the run is plain text', () => {
  const onGenerate = vi.fn()
  const { rerender } = render(<PromptBar mode="real" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  rerender(<PromptBar mode="sim" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'Hi' } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenLastCalledWith('Hi', expect.anything(), { kind: 'text' })
})

test('a text example sets Continue text; the chat example sets Chat with the default system prompt', () => {
  const onGenerate = vi.fn()
  const examples = [{ id: 't', label: 'Text', prompt: 'one two', hint: 'h' }, CHAT_EXAMPLE]
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={onGenerate} busy={false} examples={examples} />)
  fireEvent.click(screen.getAllByTestId('example-chip')[0])
  expect(onGenerate).toHaveBeenLastCalledWith('one two', expect.anything(), { kind: 'text' })
  expect(screen.getByTestId('format-text')).toHaveAttribute('aria-pressed', 'true')
  fireEvent.click(screen.getAllByTestId('example-chip')[1])
  expect(screen.getByTestId('format-chat')).toHaveAttribute('aria-pressed', 'true')
  expect(onGenerate).toHaveBeenLastCalledWith('What is the capital of France?', expect.anything(),
    { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' })
  expect(screen.getAllByTestId('example-chip')[1].querySelector('[data-testid="example-chip-tag"]')).toHaveTextContent('Chat')
})

test('the chat example in simulated mode still asks for a chat run (the app loads the model first)', () => {
  const onGenerate = vi.fn()
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={onGenerate} busy={false} examples={[CHAT_EXAMPLE]} />)
  fireEvent.click(screen.getByTestId('example-chip'))
  expect(onGenerate).toHaveBeenLastCalledWith('What is the capital of France?', expect.anything(),
    expect.objectContaining({ kind: 'chat' }))
})

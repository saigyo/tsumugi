import { fireEvent, render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import { ModelStatus } from './ModelStatus'

test('shows download progress', () => {
  render(<ModelStatus progress={{ file: 'model.onnx', loaded: 50, total: 100 }} device={null} error={null} onFallback={() => {}} />)
  expect(screen.getByTestId('model-progress')).toHaveTextContent('50')
})

test('shows device chip when ready', () => {
  render(<ModelStatus progress={null} device="webgpu" error={null} onFallback={() => {}} />)
  expect(screen.getByTestId('device-chip')).toHaveTextContent('webgpu')
})

test('error offers fallback to simulated', () => {
  const onFallback = vi.fn()
  render(<ModelStatus progress={null} device={null} error="download failed" onFallback={onFallback} />)
  expect(screen.getByTestId('model-error')).toHaveTextContent('download failed')
  fireEvent.click(screen.getByTestId('btn-fallback'))
  expect(onFallback).toHaveBeenCalled()
})

test('renders nothing when idle', () => {
  const { container } = render(<ModelStatus progress={null} device={null} error={null} onFallback={() => {}} />)
  expect(container).toBeEmptyDOMElement()
})

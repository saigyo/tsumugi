import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { ModelSlot } from './ModelSlot'

afterEach(cleanup)
const base = { mode: 'sim' as const, loading: false, fromCache: false, progress: null, error: null,
  bannerClosed: false, onLoad: () => {}, onClose: () => {}, onFallback: () => {} }

test('simulated: the banner explains the simulation and offers to load the real model', () => {
  const onLoad = vi.fn(), onClose = vi.fn()
  render(<ModelSlot {...base} onLoad={onLoad} onClose={onClose} />)
  expect(screen.getByTestId('sim-banner')).toHaveTextContent('You are watching a simulation.')
  expect(screen.getByTestId('sim-banner')).toHaveTextContent('250 MB, downloaded once, runs entirely in your browser.')
  fireEvent.click(screen.getByTestId('banner-load'))
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect(onLoad).toHaveBeenCalledOnce()
  expect(onClose).toHaveBeenCalledOnce()
})

test('closed banner or real mode: the slot is empty', () => {
  const { container } = render(<ModelSlot {...base} bannerClosed />)
  expect(container).toBeEmptyDOMElement()
  cleanup()
  const real = render(<ModelSlot {...base} mode="real" />)
  expect(real.container).toBeEmptyDOMElement()
})

test('loading shows the file and percentage even after the banner was closed', () => {
  render(<ModelSlot {...base} mode="real" loading bannerClosed progress={{ file: 'onnx/model_quantized.onnx', loaded: 42, total: 100 }} />)
  expect(screen.getByTestId('model-progress')).toHaveTextContent('Loading the real model: onnx/model_quantized.onnx · 42 %')
  expect(screen.getByTestId('model-progress-bar')).toHaveAttribute('value', '42')
})

test('loading from the cache says so', () => {
  render(<ModelSlot {...base} mode="real" loading fromCache />)
  expect(screen.getByTestId('model-progress')).toHaveTextContent('Loading the real model from the browser’s cache…')
})

test('an error shows there with the fallback button', () => {
  const onFallback = vi.fn()
  render(<ModelSlot {...base} error="download failed" onFallback={onFallback} />)
  expect(screen.getByTestId('model-error')).toHaveTextContent('Real model unavailable: download failed')
  fireEvent.click(screen.getByTestId('btn-fallback'))
  expect(onFallback).toHaveBeenCalledOnce()
})

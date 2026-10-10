import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { Formula } from './Formula'
import { texNumber } from './tex'

afterEach(() => { cleanup(); vi.restoreAllMocks() })

test('renders LaTeX as a displayed MathML equation', () => {
  render(<Formula tex="p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}" testId="f" />)
  const math = screen.getByTestId('f').querySelector('math')!
  expect(math).not.toBeNull()
  expect(math.getAttribute('display')).toBe('block')
  expect(math.querySelector('mfrac')).not.toBeNull()
})

test('inline formulas are not displayed as a block', () => {
  render(<Formula tex="x_i" inline testId="f" />)
  expect(screen.getByTestId('f').querySelector('math')!.getAttribute('display')).not.toBe('block')
})

test('invalid LaTeX shows the source as a marked fallback instead of throwing', () => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
  render(<Formula tex="\frac{1}{" testId="f" />)
  const fallback = screen.getByTestId('f')
  expect(fallback).toHaveAttribute('data-formula-error')
  expect(fallback).toHaveTextContent('\\frac{1}{')
  expect(fallback.querySelector('math')).toBeNull()
})

test('texNumber groups thousands with a thin space', () => {
  expect(texNumber(49152)).toBe('49\\,152')
  expect(texNumber(576)).toBe('576')
})

test('card formulas: shapes on a second row, indices in the softmax fraction, norm in the readout', async () => {
  const { logitsTex, softmaxTex } = await import('./tex')
  const l = logitsTex(576, 49152)
  expect(l).toContain('\\operatorname{norm}(x_{\\text{last}}) \\cdot E^{\\top}')
  expect(l).toMatch(/E\^\{\\top\} \\\\[\s\S]*\[1 \\times 576\]/)
  const s = softmaxTex(10, 49152)
  expect(s).toContain('\\exp(\\text{logit}_i / T)')
  expect(s).toContain('\\sum_j \\exp(\\text{logit}_j / T)')
  expect(s).toMatch(/\} \\\\[\s\S]*10 \\text\{ of \} 49\\,152/)
})

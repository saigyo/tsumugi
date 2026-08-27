import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { makeFixtureTrace } from '../test/fixtures'
import { DetailPanel } from './DetailPanel'

const trace = makeFixtureTrace()  // cursor 1=tokenize, 2=embed, 3..5=layers, 6=attention

afterEach(() => cleanup())

test('tokenizer detail shows token chips with ids', () => {
  render(<DetailPanel events={trace} cursor={1} mode="sim" />)
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('The')
  expect(screen.getByTestId('detail-tokenizer')).toHaveTextContent('10')  // token id
})

test('embeddings detail shows dims caption and heat cells', () => {
  render(<DetailPanel events={trace} cursor={2} mode="sim" />)
  expect(screen.getByTestId('detail-embeddings')).toHaveTextContent('576')
})

test('layers detail lights the active layer; sim shows norms', () => {
  render(<DetailPanel events={trace} cursor={4} mode="sim" />)  // layer index 1 of 3
  const blocks = screen.getAllByTestId('layer-block')
  expect(blocks).toHaveLength(3)
  expect(blocks[1].dataset.lit).toBe('true')
  expect(blocks[2].dataset.lit).toBe('false')
})

test('real mode labels layers as schematic', () => {
  render(<DetailPanel events={trace} cursor={4} mode="real" />)
  expect(screen.getByTestId('detail-layers')).toHaveTextContent(/schematic/i)
})

test('no relevant event renders empty state', () => {
  render(<DetailPanel events={trace} cursor={-1} mode="sim" />)
  expect(screen.getByTestId('detail-empty')).toBeInTheDocument()
})

test('truncated tokenize event shows a notice', () => {
  const t = makeFixtureTrace()
  const tok = t[1]
  if (tok.type === 'tokenize') tok.truncated = true
  render(<DetailPanel events={t} cursor={1} mode="sim" />)
  expect(screen.getByTestId('truncation-notice')).toBeInTheDocument()
})

// makeFixtureTrace cycle 0: index 7=logits, 8=softmax, 9=sample, 10=append
test('logits detail shows one bar per candidate', () => {
  render(<DetailPanel events={trace} cursor={7} mode="sim" />)
  expect(screen.getAllByTestId('logit-bar')).toHaveLength(3)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('sat')
})

test('softmax cursor switches bars to probabilities', () => {
  render(<DetailPanel events={trace} cursor={8} mode="sim" />)
  expect(screen.getByTestId('detail-logits')).toHaveTextContent('70')  // 0.7 → 70%
})

test('sampler detail marks the chosen token', () => {
  render(<DetailPanel events={trace} cursor={9} mode="sim" />)
  expect(screen.getByTestId('chosen-marker')).toHaveTextContent('sat')
})

test('run-end cursor shows a run summary, not the idle hint', () => {
  render(<DetailPanel events={trace} cursor={trace.length - 1} mode="sim" />)
  const summary = screen.getByTestId('detail-run-end')
  expect(summary).toHaveTextContent(/max-tokens/)
  expect(summary).toHaveTextContent('2')          // generated token count
  expect(summary).toHaveTextContent(/T=0.8/)      // params from run-start
  expect(screen.queryByTestId('detail-empty')).not.toBeInTheDocument()
})

test('attention cursor maps to layers stage and shows the heatmap', () => {
  render(<DetailPanel events={trace} cursor={6} mode="sim" />)
  expect(screen.getByTestId('detail-layers')).toBeInTheDocument()
  expect(screen.getByTestId('attention-heatmap')).toBeInTheDocument()
})

test('mid-layer cursor before the cycle attention has no heatmap', () => {
  render(<DetailPanel events={trace} cursor={4} mode="sim" />)
  expect(screen.queryByTestId('attention-heatmap')).not.toBeInTheDocument()
})

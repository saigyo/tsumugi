import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { GateUpScatter, ProductBars, SiluChart, StepStrip } from './MlpStepCharts'
import { CHART_W, gateX } from './stepChartMath'
import { productSummary } from '../neurons/steps'

afterEach(cleanup)
const gate = Float32Array.from([-3, -0.5, 0.2, 6, 11])
const up = Float32Array.from([1, -2, 0.5, -1.5, 2])
const h = Float32Array.from(gate, (g, i) => (g / (1 + Math.exp(-g))) * up[i])

test('every chart is a full-width SVG on the shared 720-unit frame', () => {
  render(<>
    <GateUpScatter gate={gate} up={up} domain={[-4, 12]} marked={[4]} named={[4]} hover={null} onHover={() => {}} box={null} />
    <SiluChart gate={gate} domain={[-4, 12]} marked={[4]} counts={{ negative: 2, positive: 3 }} hover={null} onHover={() => {}} box={null} />
    <ProductBars h={h} summary={productSummary(h)} marked={[4]} hover={null} onHover={() => {}} box={null} />
  </>)
  for (const id of ['steps-scatter', 'steps-silu', 'steps-product']) {
    const svg = screen.getByTestId(id)
    expect(svg.getAttribute('viewBox')).toMatch(new RegExp(`^0 0 ${CHART_W} \\d+$`))
    expect(svg).toHaveClass('mlp-chart')
  }
})

test('scatter and SiLU place a neuron at the same x', () => {
  render(<>
    <GateUpScatter gate={gate} up={up} domain={[-4, 12]} marked={[]} named={[]} hover={null} onHover={() => {}} box={null} />
    <SiluChart gate={gate} domain={[-4, 12]} marked={[]} counts={{ negative: 2, positive: 3 }} hover={null} onHover={() => {}} box={null} />
  </>)
  const x = (id: string) => screen.getByTestId(id).querySelector('[data-neuron="3"]')!.getAttribute('cx')
  expect(x('steps-scatter')).toBe(x('steps-silu'))
  expect(Number(x('steps-scatter'))).toBeCloseTo(gateX([-4, 12])(6), 3)
})

test('marked neurons are drawn in the accent and are focusable; others are not', () => {
  render(<GateUpScatter gate={gate} up={up} domain={[-4, 12]} marked={[4, 1]} named={[4]} hover={null} onHover={() => {}} box={null} />)
  const dot = (i: number) => screen.getByTestId('steps-scatter').querySelector(`[data-neuron="${i}"]`)!
  expect(dot(4)).toHaveClass('step-dot-marked')
  expect(dot(4).getAttribute('tabindex')).toBe('0')
  expect(dot(0).getAttribute('tabindex')).toBeNull()
  expect(screen.getByTestId('steps-scatter')).toHaveTextContent('N4')
})

test('hovering a dot or bar reports the neuron and the chart', () => {
  const onHover = vi.fn()
  render(<ProductBars h={h} summary={productSummary(h)} marked={[]} hover={null} onHover={onHover} box={null} />)
  fireEvent.mouseEnter(screen.getByTestId('steps-product').querySelector('[data-neuron="4"]')!)
  expect(onHover).toHaveBeenCalledWith(expect.objectContaining({ neuron: 4, chart: 'product', height: 150 }))
})

test('the product chart marks the half and 90 % counts and names what it leaves out', () => {
  const many = Float32Array.from({ length: 200 }, (_, i) => 10 / (i + 1))
  render(<ProductBars h={many} summary={productSummary(many)} marked={[]} hover={null} onHover={() => {}} box={null} />)
  const svg = screen.getByTestId('steps-product')
  expect(svg.querySelectorAll('[data-neuron]')).toHaveLength(120)
  expect(svg).toHaveTextContent('1 neuron carries half')
  expect(svg).toHaveTextContent(/the other 80 are all below 0\.08/)
})

test('a strip stretches one cell per value across the full width', () => {
  render(<StepStrip values={Float32Array.from([1, -1, 0])} lim={1} label="x, into the MLP" testId="strip-x" />)
  const svg = screen.getByTestId('strip-x')
  expect(svg.getAttribute('viewBox')).toBe('0 0 3 1')
  expect(svg.getAttribute('preserveAspectRatio')).toBe('none')
  expect(svg.querySelectorAll('rect')).toHaveLength(3)
  expect(screen.getByText('x, into the MLP')).toBeInTheDocument()
})

test('a marked bar is focusable, named, and reports its neuron on focus', () => {
  const onHover = vi.fn()
  render(<ProductBars h={h} summary={productSummary(h)} marked={[4]} hover={null} onHover={onHover} box={null} />)
  const bar = screen.getByTestId('steps-product').querySelector('[data-neuron="4"]')!
  expect(bar.getAttribute('tabindex')).toBe('0')
  expect(bar.getAttribute('aria-label')).toBe('N4')
  fireEvent.focus(bar)
  expect(onHover).toHaveBeenCalledWith(expect.objectContaining({ neuron: 4, chart: 'product' }))
})

test('one neuron carrying half is named in the singular', () => {
  const one = Float32Array.from([100, 1, 1, 1, 1])
  render(<ProductBars h={one} summary={productSummary(one)} marked={[]} hover={null} onHover={() => {}} box={null} />)
  expect(screen.getByTestId('steps-product')).toHaveTextContent('1 neuron carries half')
  expect(screen.getByTestId('steps-product')).toHaveTextContent('1 neuron carries 90 %')
})

test('a count beyond the bars shown gets an off-chart note at the right end', () => {
  const flat = Float32Array.from({ length: 500 }, () => 1)
  render(<ProductBars h={flat} summary={productSummary(flat)} marked={[]} hover={null} onHover={() => {}} box={null} />)
  const svg = screen.getByTestId('steps-product')
  expect(svg.querySelectorAll('.step-cut')).toHaveLength(0)
  const notes = [...svg.querySelectorAll('text')].filter((t) => /needs/.test(t.textContent!))
  expect(notes.map((t) => t.textContent)).toEqual(['half needs 250 neurons', '90 % needs 450 neurons'])
  for (const t of notes) expect(t.getAttribute('text-anchor')).toBe('end')
})

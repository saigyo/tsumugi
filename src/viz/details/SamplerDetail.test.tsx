import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { buildFixtureTrace } from '../../test/fixtures'
import type { TraceEvent } from '../../trace/types'
import { TokenPopover } from '../TokenPopover'
import { SamplerDetail } from './SamplerDetail'

afterEach(cleanup)

const trace = buildFixtureTrace({ cycles: 1, chosen: [{ id: 198, text: '\n' }] })
const softmax = trace.find((e) => e.type === 'softmax') as Extract<TraceEvent, { type: 'softmax' }>
const sample = trace.find((e) => e.type === 'sample') as Extract<TraceEvent, { type: 'sample' }>

test('the chosen line shows a newline token as ↵, as a chosen chip followed by its id, without quotation marks', () => {
  render(<SamplerDetail softmax={softmax} sample={sample} />)
  const line = screen.getByTestId('chosen-marker')
  expect(line).toHaveTextContent('chose ↵ (id 198)')
  expect(line.textContent).not.toMatch(/["“”]/)
  const chip = line.querySelector('.tok')!
  expect(chip).toHaveClass('is-chosen')
  expect(chip).toHaveTextContent('↵')
})

test('the token popover shows whitespace tokens as glyphs', () => {
  render(<TokenPopover softmax={softmax} sample={sample} />)
  expect(screen.getAllByTestId('popover-row').map((r) => r.querySelector('.popover-token')!.textContent))
    .toEqual(['↵', '␣ran', '␣was'])
  const rows = screen.getAllByTestId('popover-row')
  expect(rows.every((r) => r.querySelector('.tok.tok-sm') !== null)).toBe(true)
  expect(rows.map((r) => r.querySelector('.bar')!.classList.contains('bar-chosen'))).toEqual(rows.map((r) => r.dataset.chosen === 'true'))
})

test('a forced sample says it was copied, not drawn', () => {
  render(<SamplerDetail softmax={softmax} sample={{ ...sample, method: 'forced' }} />)
  expect(screen.getByTestId('detail-sampler')).toHaveTextContent('Copied from the original run (knockout replay): no draw.')
  expect(screen.getByTestId('detail-sampler')).not.toHaveTextContent('A weighted draw')
})

test('a drawn sample keeps the weighted-draw sentence', () => {
  render(<SamplerDetail softmax={softmax} sample={sample} />)
  expect(screen.getByTestId('detail-sampler')).toHaveTextContent('A weighted draw across the probability strip')
})

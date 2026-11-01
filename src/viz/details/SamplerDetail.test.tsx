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

test('the chosen line shows a newline token as ↵', () => {
  render(<SamplerDetail softmax={softmax} sample={sample} />)
  expect(screen.getByTestId('chosen-marker')).toHaveTextContent('chose "↵" (id 198)')
})

test('the token popover shows whitespace tokens as glyphs', () => {
  render(<TokenPopover softmax={softmax} sample={sample} />)
  expect(screen.getAllByTestId('popover-row').map((r) => r.querySelector('.popover-token')!.textContent))
    .toEqual(['↵', '␣ran', '␣was'])
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

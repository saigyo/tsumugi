import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { fixtureVector, makeGeometryAsset } from '../test/geometryFixture'
import type { TokenInfo } from '../trace/types'
import { EmbeddingGeometry } from './EmbeddingGeometry'

afterEach(() => cleanup())
const asset = makeGeometryAsset()
const toks = (ids: number[]): TokenInfo[] => ids.map((id) => ({ id, text: ` t${id}` }))
const vecFor = (tokens: TokenInfo[]) => (pos: number) => fixtureVector(tokens[pos].id)
const noop = () => {}

test('lists 8 renderable neighbours of the selected token with similarities', () => {
  const tokens = toks([6, 20])
  render(<EmbeddingGeometry tokens={tokens} selected={0} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  const rows = screen.getAllByTestId('embed-neighbor')
  expect(rows).toHaveLength(8)
  expect(rows[0]).toHaveTextContent('t5')          // ids 7 and 8 are unrenderable and skipped
  expect(screen.getByTestId('embed-neighbors')).not.toHaveTextContent('<0x07>')
  expect(rows[0]).toHaveTextContent(/0\.9\d|1\.00/)
})

test('similarity matrix has n² cells and no cap note for short sequences', () => {
  const tokens = toks([1, 2, 130])
  render(<EmbeddingGeometry tokens={tokens} selected={2} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  expect(screen.getAllByTestId('sim-cell')).toHaveLength(9)
  expect(screen.queryByTestId('embed-sim-cap')).toBeNull()
})

test('matrix is capped at the last 24 tokens with a note', () => {
  const tokens = toks(Array.from({ length: 30 }, (_, i) => i + 40))
  render(<EmbeddingGeometry tokens={tokens} selected={29} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  expect(screen.getAllByTestId('sim-cell')).toHaveLength(24 * 24)
  expect(screen.getByTestId('embed-sim-cap')).toHaveTextContent('last 24 of 30')
})

test('error state shows the message and retry calls back; no matrix without vectors', () => {
  const retry = vi.fn()
  render(<EmbeddingGeometry tokens={toks([1])} selected={0} vectorFor={() => undefined}
    loading={false} error="offline" retry={retry} source="asset" />)
  expect(screen.getByTestId('embed-geometry-error')).toHaveTextContent("Vocabulary geometry couldn't be loaded")
  fireEvent.click(screen.getByTestId('embed-geometry-retry'))
  expect(retry).toHaveBeenCalledTimes(1)
  expect(screen.queryByTestId('embed-similarity')).toBeNull()
  expect(screen.queryByTestId('embed-neighbors')).toBeNull()
  expect(screen.queryByTestId('embed-provenance')).toBeNull()
})

test('loading state', () => {
  render(<EmbeddingGeometry tokens={toks([1])} selected={0} vectorFor={() => undefined}
    loading={true} retry={noop} source="asset" />)
  expect(screen.getByTestId('embed-geometry-loading')).toBeInTheDocument()
})

test('provenance caption follows the source', () => {
  const tokens = toks([1, 2])
  const { unmount } = render(<EmbeddingGeometry tokens={tokens} selected={0} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="model" />)
  expect(screen.getByTestId('embed-provenance')).toHaveTextContent('Exact rows from the running model.')
  unmount()
  render(<EmbeddingGeometry tokens={tokens} selected={0} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  expect(screen.getByTestId('embed-provenance')).toHaveTextContent('reduced to 64 dimensions offline; similarities are approximate.')
})

test('the similarity colour scale is stretched to the matrix\'s own off-diagonal range', () => {
  // ids 1, 2 are near-identical directions; 130 is the antipode of 2 → the range is wide
  const tokens = toks([1, 2, 130])
  render(<EmbeddingGeometry tokens={tokens} selected={0} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  const cells = screen.getAllByTestId('sim-cell')
  const fill = (i: number) => cells[i].getAttribute('fill') ?? ''
  const lightness = (f: string) => Number(/(\d+)%\)$/.exec(f)?.[1])
  expect(lightness(fill(0))).toBe(24)                           // diagonal (1.0) is the darkest step
  expect(lightness(fill(1))).toBeLessThan(lightness(fill(2)))    // (1,2) similar → darker than (1,130)
  expect(lightness(fill(2))).toBe(94)                           // the minimum maps to the lightest step
  const legend = screen.getByTestId('embed-sim-legend')
  expect(legend).toHaveTextContent(/colour range/)
  expect(legend).toHaveTextContent(/-?\d\.\d\d … \d\.\d\d/)
})

test('a leading space in a token is shown as a visible marker, with a note explaining it', () => {
  const tokens: TokenInfo[] = [{ id: 6, text: ' t6' }]
  render(<EmbeddingGeometry tokens={tokens} selected={0} vectorFor={vecFor(tokens)} asset={asset}
    loading={false} retry={noop} source="asset" />)
  expect(screen.getByTestId('embed-neighbors')).toHaveTextContent('Nearest to ␣t6')
  // fixture texts carry no leading space, so neighbour labels stay bare
  expect(screen.getAllByTestId('embed-neighbor')[0]).toHaveTextContent(/^t5/)
  expect(screen.getByTestId('embed-space-note').getAttribute('title')).toMatch(/leading space/)
})

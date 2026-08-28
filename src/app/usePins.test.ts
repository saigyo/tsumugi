import { act, renderHook } from '@testing-library/react'
import { expect, test } from 'vitest'
import type { HeadData } from '../engine/transformers/TransformersEngine'
import { usePins } from './usePins'

const data = (layer: number, head: number, extra?: Partial<HeadData>): HeadData => ({
  layer, head, matrix: [[1], [0.5, 0.5]], label: null, score: null, ...extra,
})

test('pin fetches and appends with the pinned fallback label', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h)))
  await act(() => result.current.pin(3, 1))
  expect(result.current.pins).toHaveLength(1)
  expect(result.current.pins[0]).toMatchObject({ layer: 3, head: 1, label: 'pinned' })
  expect(result.current.pins[0].matrix).toEqual([[1], [0.5, 0.5]])
  expect(result.current.note).toBeNull()
})

test('a resolved template label and score are kept', async () => {
  const { result } = renderHook(() =>
    usePins(async (l, h) => data(l, h, { label: 'previous-token', score: 0.8 })))
  await act(() => result.current.pin(0, 0))
  expect(result.current.pins[0]).toMatchObject({ label: 'previous-token', score: 0.8 })
})

test('duplicate pins are ignored', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h)))
  await act(() => result.current.pin(1, 1))
  await act(() => result.current.pin(1, 1))
  expect(result.current.pins).toHaveLength(1)
})

test('an empty matrix sets the stale note and adds nothing', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h, { matrix: [] })))
  await act(() => result.current.pin(0, 0))
  expect(result.current.pins).toHaveLength(0)
  expect(result.current.note).toBe('run data no longer available — regenerate to explore heads')
})

test('the sixth pin evicts the first (FIFO cap of 5)', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h)))
  for (let h = 0; h < 6; h++) await act(() => result.current.pin(0, h))
  expect(result.current.pins).toHaveLength(5)
  expect(result.current.pins.map((p) => p.head)).toEqual([1, 2, 3, 4, 5])
})

test('reset clears pins and note', async () => {
  const { result } = renderHook(() => usePins(async (l, h) => data(l, h)))
  await act(() => result.current.pin(0, 0))
  act(() => result.current.reset())
  expect(result.current.pins).toHaveLength(0)
  expect(result.current.note).toBeNull()
})

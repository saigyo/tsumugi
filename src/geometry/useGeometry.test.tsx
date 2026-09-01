import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test } from 'vitest'
import { stubGeometryFetch } from '../test/geometryFixture'
import { resetGeometryCache } from './asset'
import { useGeometry } from './useGeometry'

beforeEach(() => resetGeometryCache())
afterEach(() => resetGeometryCache())

test('loads on mount and reports ready with the asset', async () => {
  stubGeometryFetch()
  const { result } = renderHook(() => useGeometry())
  expect(result.current.status).toBe('loading')
  await waitFor(() => expect(result.current.status).toBe('ready'))
  expect(result.current.asset?.text(3)).toBe('t3')
})

test('reports error with the message, and retry re-fetches', async () => {
  stubGeometryFetch({ 'manifest.json': () => { throw new Error('offline') } })
  const { result } = renderHook(() => useGeometry())
  await waitFor(() => expect(result.current.status).toBe('error'))
  expect(result.current.error).toBe('offline')
  const ok = stubGeometryFetch()
  act(() => result.current.retry())
  await waitFor(() => expect(result.current.status).toBe('ready'))
  expect(ok).toHaveBeenCalledTimes(4)
})

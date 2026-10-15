import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { fixtureNeuronAsset, fixtureNeuronLayer } from '../test/neuronFixture'
import * as asset from './asset'
import { useNeuronLayer } from './useNeuronLayer'

afterEach(() => vi.restoreAllMocks())

test('idle without a layer, then loading, then ready', async () => {
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  vi.spyOn(asset, 'loadNeuronLayer').mockImplementation(async (l) => fixtureNeuronLayer(l))
  const { result, rerender } = renderHook(({ l }) => useNeuronLayer(l), { initialProps: { l: null as number | null } })
  expect(result.current.status).toBe('idle')
  rerender({ l: 1 })
  expect(result.current.status).toBe('loading')
  await waitFor(() => expect(result.current.status).toBe('ready'))
})

test('a stale result for an earlier layer never replaces the current one', async () => {
  let release0: () => void = () => {}
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  vi.spyOn(asset, 'loadNeuronLayer').mockImplementation((l) => l === 0
    ? new Promise((resolve) => { release0 = () => resolve(fixtureNeuronLayer(0)) })
    : Promise.resolve(fixtureNeuronLayer(l)))
  const { result, rerender } = renderHook(({ l }) => useNeuronLayer(l), { initialProps: { l: 0 as number | null } })
  rerender({ l: 2 })
  await waitFor(() => expect(result.current.status).toBe('ready'))
  const shown = result.current.status === 'ready' ? result.current.layer : null
  await act(async () => { release0() })
  expect(result.current.status === 'ready' && result.current.layer).toBe(shown)
})

test('an error offers retry, which loads again', async () => {
  vi.spyOn(asset, 'loadNeuronAsset').mockResolvedValue(fixtureNeuronAsset())
  const load = vi.spyOn(asset, 'loadNeuronLayer').mockRejectedValueOnce(new Error('HTTP 503'))
    .mockImplementation(async (l) => fixtureNeuronLayer(l))
  const { result } = renderHook(() => useNeuronLayer(0))
  await waitFor(() => expect(result.current.status).toBe('error'))
  act(() => result.current.retry())
  await waitFor(() => expect(result.current.status).toBe('ready'))
  expect(load).toHaveBeenCalledTimes(2)
})

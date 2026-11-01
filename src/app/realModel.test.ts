import { afterEach, expect, test, vi } from 'vitest'
import { MODEL_FILE_URL, readFlag, realModelCached, writeFlag } from './realModel'

afterEach(() => { localStorage.clear(); vi.restoreAllMocks() })

test('flags round-trip through localStorage', () => {
  expect(readFlag('tsumugi.realModelUsed')).toBe(false)
  writeFlag('tsumugi.realModelUsed', true)
  expect(readFlag('tsumugi.realModelUsed')).toBe(true)
  writeFlag('tsumugi.realModelUsed', false)
  expect(readFlag('tsumugi.realModelUsed')).toBe(false)
})

test('storage that throws reads as unset and never throws', () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
  expect(readFlag('tsumugi.simBannerClosed')).toBe(false)
  expect(() => writeFlag('tsumugi.simBannerClosed', true)).not.toThrow()
})

test('the model counts as cached only when its file is in the transformers cache', async () => {
  const fake = (has: boolean, file: boolean) => ({
    has: async () => has,
    open: async () => ({ match: async (url: string) => (file && url === MODEL_FILE_URL ? new Response('x') : undefined) }),
  }) as unknown as CacheStorage
  expect(MODEL_FILE_URL).toBe('https://huggingface.co/saigyo-hoshi/smollm2-135m-attn-onnx/resolve/v7/onnx/model_quantized.onnx')
  expect(await realModelCached(fake(true, true))).toBe(true)
  expect(await realModelCached(fake(true, false))).toBe(false)
  expect(await realModelCached(fake(false, false))).toBe(false)
  expect(await realModelCached(undefined)).toBe(false)
})

import { create } from 'zustand'
import { ATTN_MODEL_ID, ATTN_MODEL_REVISION } from '../engine/tokenizer'

// Any card can ask the app to load the real model; the app installs the
// handler and keeps loading up to date.
interface RealModelState { request: () => void; loading: boolean }
export const useRealModel = create<RealModelState>(() => ({ request: () => {}, loading: false }))

export type PrefKey = 'tsumugi.simBannerClosed' | 'tsumugi.realModelUsed'

// Per-browser conveniences: without storage the app behaves as on a first visit.
export function readFlag(key: PrefKey): boolean {
  try { return localStorage.getItem(key) === '1' } catch { return false }
}
export function writeFlag(key: PrefKey, on: boolean): void {
  try { if (on) localStorage.setItem(key, '1'); else localStorage.removeItem(key) } catch { /* session only */ }
}

// transformers.js caches each file under its download URL in the Cache
// Storage named by env.cacheKey ('transformers-cache').
// Only the custom export counts: a browser that only has the stock fallback
// starts simulated, and its next load tries the custom export again.
export const MODEL_FILE_URL =
  `https://huggingface.co/${ATTN_MODEL_ID}/resolve/${ATTN_MODEL_REVISION}/onnx/model_quantized.onnx`

export async function realModelCached(
  store: CacheStorage | undefined = typeof caches === 'undefined' ? undefined : caches,
): Promise<boolean> {
  if (!store) return false
  try {
    if (!(await store.has('transformers-cache'))) return false
    return (await (await store.open('transformers-cache')).match(MODEL_FILE_URL)) !== undefined
  } catch { return false }
}

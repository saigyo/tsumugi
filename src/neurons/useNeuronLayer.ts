import { useCallback, useEffect, useState } from 'react'
import { loadNeuronAsset, loadNeuronLayer, type NeuronAsset } from './asset'
import type { NeuronLayer } from './format'

export type NeuronLayerState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; asset: NeuronAsset; layer: NeuronLayer }
  | { status: 'error'; error: string }

// Loads the asset and one layer file. Results are keyed by (layer, attempt):
// a result for any other key is ignored, so a slow earlier layer never
// replaces the one on screen, and a new key reads as 'loading' at once
// (no state set inside the effect).
export function useNeuronLayer(layer: number | null): NeuronLayerState & { retry(): void } {
  const [attempt, setAttempt] = useState(0)
  const [result, setResult] = useState<{ key: string; state: NeuronLayerState } | null>(null)
  const key = layer === null ? null : `${layer}:${attempt}`
  useEffect(() => {
    if (layer === null) return
    let live = true
    const k = `${layer}:${attempt}`
    Promise.all([loadNeuronAsset(), loadNeuronLayer(layer)]).then(
      ([asset, data]) => { if (live) setResult({ key: k, state: { status: 'ready', asset, layer: data } }) },
      (err: unknown) => { if (live) setResult({ key: k, state: { status: 'error', error: err instanceof Error ? err.message : String(err) } }) },
    )
    return () => { live = false }
  }, [layer, attempt])
  const retry = useCallback(() => setAttempt((n) => n + 1), [])
  const state: NeuronLayerState = key === null ? { status: 'idle' }
    : result?.key === key ? result.state : { status: 'loading' }
  return { ...state, retry }
}

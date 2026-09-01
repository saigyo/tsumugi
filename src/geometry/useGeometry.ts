import { useCallback, useEffect, useState } from 'react'
import { loadGeometry, type GeometryAsset } from './asset'

export type GeometryStatus = 'idle' | 'loading' | 'ready' | 'error'
export interface GeometryState {
  status: GeometryStatus
  asset?: GeometryAsset
  error?: string
  retry(): void
}

// Kicks off the shared, once-per-page asset load on first mount; `retry`
// re-runs it after a failure (loadGeometry drops a rejected promise, so the
// retry really re-fetches).
export function useGeometry(): GeometryState {
  const [state, setState] = useState<Omit<GeometryState, 'retry'>>({ status: 'idle' })
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let live = true
    setState({ status: 'loading' })
    loadGeometry().then(
      (asset) => { if (live) setState({ status: 'ready', asset }) },
      (err: unknown) => { if (live) setState({ status: 'error', error: err instanceof Error ? err.message : String(err) }) },
    )
    return () => { live = false }
  }, [attempt])
  const retry = useCallback(() => setAttempt((n) => n + 1), [])
  return { ...state, retry }
}

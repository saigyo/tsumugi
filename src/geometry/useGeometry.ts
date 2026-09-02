import { useCallback, useEffect, useState } from 'react'
import { loadGeometry, type GeometryAsset } from './asset'

export type GeometryStatus = 'loading' | 'ready' | 'error'
export interface GeometryState {
  status: GeometryStatus
  asset?: GeometryAsset
  error?: string
  retry(): void
}

// Kicks off the shared, once-per-page asset load on first mount; `retry`
// re-runs it after a failure (loadGeometry drops a rejected promise, so the
// retry really re-fetches). The hook always fetches on mount, so state starts
// in 'loading' directly rather than routing through an idle state; `retry`
// sets 'loading' itself (from the event handler, not the effect) before
// bumping `attempt` to re-run the effect.
export function useGeometry(): GeometryState {
  const [state, setState] = useState<Omit<GeometryState, 'retry'>>({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let live = true
    loadGeometry().then(
      (asset) => { if (live) setState({ status: 'ready', asset }) },
      (err: unknown) => { if (live) setState({ status: 'error', error: err instanceof Error ? err.message : String(err) }) },
    )
    return () => { live = false }
  }, [attempt])
  const retry = useCallback(() => {
    setState({ status: 'loading' })
    setAttempt((n) => n + 1)
  }, [])
  return { ...state, retry }
}

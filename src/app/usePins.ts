import { useCallback, useRef, useState } from 'react'
import type { HeadData } from '../engine/transformers/TransformersEngine'
import type { AttentionHead } from '../trace/types'

const STALE_NOTE = 'run data no longer available — regenerate to explore heads'
const MAX_PINS = 5

// Pinned heads are run-scoped UI state: App resets them when a new run
// replaces the trace (they must survive scrubbing away from the Layers stage,
// so they cannot live inside LayersDetail). A ref mirrors the pin list so the
// duplicate check reads current state synchronously — a setState-updater
// read is not guaranteed to run before the next line.
export function usePins(fetchHead: (layer: number, head: number) => Promise<HeadData>) {
  const [pins, setPins] = useState<AttentionHead[]>([])
  const [note, setNote] = useState<string | null>(null)
  const pinsRef = useRef<AttentionHead[]>([])
  const fetchRef = useRef(fetchHead)
  fetchRef.current = fetchHead

  const commit = useCallback((next: AttentionHead[]) => {
    pinsRef.current = next
    setPins(next)
  }, [])

  const pin = useCallback(async (layer: number, head: number) => {
    if (pinsRef.current.some((p) => p.layer === layer && p.head === head)) return
    const r = await fetchRef.current(layer, head)
    if (r.matrix.length === 0) { setNote(STALE_NOTE); return }
    if (pinsRef.current.some((p) => p.layer === layer && p.head === head)) return
    setNote(null)
    const next: AttentionHead = { layer, head, label: r.label ?? 'pinned',
      ...(r.score != null ? { score: r.score } : {}), matrix: r.matrix }
    commit([...pinsRef.current, next].slice(-MAX_PINS))
  }, [commit])

  const reset = useCallback(() => { commit([]); setNote(null) }, [commit])

  return { pins, note, pin, reset }
}

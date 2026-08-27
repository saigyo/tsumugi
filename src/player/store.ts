import { useEffect } from 'react'
import { create } from 'zustand'
import { useTraceStore } from '../trace/store'
import { delayFor } from './pacing'
import { initialPlayerState, playerReducer, type PlayerAction, type PlayerState } from './reducer'

interface PlayerStore extends PlayerState {
  dispatch: (a: PlayerAction) => void
}

export const usePlayerStore = create<PlayerStore>()((set) => ({
  ...initialPlayerState,
  dispatch: (a) => set((s) => playerReducer(s, a)),
}))

export function usePlaybackTicker(): void {
  const status = usePlayerStore((s) => s.status)
  const speed = usePlayerStore((s) => s.speed)
  const cursor = usePlayerStore((s) => s.cursor)
  const length = useTraceStore((s) => s.events.length)

  useEffect(() => {
    if (status !== 'playing' || length === 0) return
    if (cursor >= length - 1) {
      // parked at the frontier: a finished trace ends playback; a growing one waits
      if (useTraceStore.getState().events[cursor]?.type === 'run-end') {
        usePlayerStore.getState().dispatch({ type: 'pause' })
      }
      return
    }
    const next = useTraceStore.getState().events[cursor + 1]
    const t = setTimeout(
      () => usePlayerStore.getState().dispatch({ type: 'stepForward', auto: true }),
      delayFor(next, speed),
    )
    return () => clearTimeout(t)
  }, [status, speed, cursor, length])
}

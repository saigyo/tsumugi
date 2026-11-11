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
  // Not the length itself: a wait that restarted with every arriving event would never end while a slow model
  // keeps writing. Only whether there is a next event matters, and whether there is anything at all.
  const empty = useTraceStore((s) => s.events.length === 0)
  const hasNext = useTraceStore((s) => cursor < s.events.length - 1)

  useEffect(() => {
    if (status !== 'playing' || empty) return
    if (!hasNext) {
      // parked at the frontier: a finished trace ends playback; a growing one waits
      if (useTraceStore.getState().events[cursor]?.type === 'run-end') {
        usePlayerStore.getState().dispatch({ type: 'pause' })
      }
      return
    }
    // A step stays on screen for its own time (not for that of the step after it: the Embeddings step, which a
    // quick layer step follows, would flash by). Before the first step there is nothing to wait for.
    const shown = cursor >= 0 ? useTraceStore.getState().events[cursor] : undefined
    const t = setTimeout(
      () => usePlayerStore.getState().dispatch({ type: 'stepForward', auto: true }),
      shown ? delayFor(shown, speed) : 0,
    )
    return () => clearTimeout(t)
  }, [status, speed, cursor, empty, hasNext])
}

import { create } from 'zustand'
import type { TraceEvent } from './types'

interface TraceState {
  events: TraceEvent[]
  append: (e: TraceEvent) => void
  clear: () => void
  load: (events: TraceEvent[]) => void
}

export const useTraceStore = create<TraceState>()((set) => ({
  events: [],
  append: (e) => set((s) => ({ events: [...s.events, e] })),
  clear: () => set({ events: [] }),
  load: (events) => set({ events }),
}))

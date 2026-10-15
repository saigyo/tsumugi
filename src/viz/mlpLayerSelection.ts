import { create } from 'zustand'
import type { TraceEvent } from '../trace/types'

// The layer "Inside one MLP" shows, set by clicks in either #15 card and by
// its own stepper. Tied to the run it was picked in (that trace's run-start
// event): a new run, another archived run or an import falls back to the
// default layer (the largest MLP vote for the selected candidate).
interface MlpLayerSelection {
  pick: { run: TraceEvent; layer: number } | null
  select: (run: TraceEvent, layer: number) => void
}

export const useMlpLayer = create<MlpLayerSelection>((set) => ({
  pick: null,
  select: (run, layer) => set({ pick: { run, layer } }),
}))

export const resetMlpLayer = () => useMlpLayer.setState({ pick: null })

export function selectedLayer(pick: MlpLayerSelection['pick'], run: TraceEvent | undefined): number | null {
  return pick && run && pick.run === run ? pick.layer : null
}

import { create } from 'zustand'
import type { TraceEvent } from '../trace/types'
import { selectedCandidate, useCandidateSelection } from './candidateSelection'
import { topMlpLayers } from './mlpMath'
import type { VotedCandidate } from './selectors'

// The layer "Inside one MLP" shows, set by clicks in either #15 card and by
// its own stepper. Tied to the run it was picked in (that trace's run-start
// event): a new run, another archived run or an import falls back to the
// default layer (the largest MLP vote for the selected candidate).
export interface MlpLayerSelection {
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

// The MLP layer the neuron card, the step-by-step card and the navigation bar
// show: this run's pick, else the layer with the largest MLP vote for the
// selected candidate. One rule, so the three always agree.
export function effectiveMlpLayer(pick: MlpLayerSelection['pick'], run: TraceEvent | undefined,
  candidates: VotedCandidate[] | undefined, selected: number): number | null {
  if (!candidates || candidates.length === 0) return null
  return selectedLayer(pick, run) ?? topMlpLayers(candidates[Math.min(selected, candidates.length - 1)].votes, 1)[0].layer
}

export function useMlpLayerControl(candidates: VotedCandidate[] | undefined, run: TraceEvent | undefined) {
  const pick = useMlpLayer((s) => s.pick)
  const select = useMlpLayer((s) => s.select)
  const candPick = useCandidateSelection((s) => s.pick)
  const selected = candidates && candidates.length > 0 ? selectedCandidate(candPick, candidates) : 0
  const layer = effectiveMlpLayer(pick, run, candidates, selected)
  const layers = candidates?.[0]?.votes.mlp.length ?? 0
  const step = (d: number) => { if (run && layer !== null) select(run, Math.min(layers - 1, Math.max(0, layer + d))) }
  return { layer, layers, selected, step }
}

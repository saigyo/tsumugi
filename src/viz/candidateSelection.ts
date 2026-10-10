import { create } from 'zustand'

// The candidate the Logits card and the votes card both highlight. Keyed by
// cycle: moving to another cycle falls back to the top candidate. A store,
// not component state, so the pick survives switching stages.
interface CandidateSelection {
  pick: { cycle: number; index: number } | null
  select: (cycle: number, index: number) => void
}

export const useCandidateSelection = create<CandidateSelection>((set) => ({
  pick: null,
  select: (cycle, index) => set({ pick: { cycle, index } }),
}))

export const resetCandidateSelection = () => useCandidateSelection.setState({ pick: null })

export function selectedCandidate(pick: CandidateSelection['pick'], cycle: number, count: number): number {
  return pick && pick.cycle === cycle && pick.index < count ? pick.index : 0
}

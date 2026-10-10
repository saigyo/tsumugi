import { create } from 'zustand'
import type { LogitCandidate } from '../trace/types'

// The candidate the Logits card and the votes card both highlight. A store,
// not component state, so the pick survives switching stages. The pick is
// tied to the candidate list it was made in (one logits event's topK, which
// both cards read): another cycle, a new run or an imported trace brings a
// different list, and the selection falls back to the top candidate.
interface CandidateSelection {
  pick: { topK: readonly LogitCandidate[]; index: number } | null
  select: (topK: readonly LogitCandidate[], index: number) => void
}

export const useCandidateSelection = create<CandidateSelection>((set) => ({
  pick: null,
  select: (topK, index) => set({ pick: { topK, index } }),
}))

export const resetCandidateSelection = () => useCandidateSelection.setState({ pick: null })

export function selectedCandidate(pick: CandidateSelection['pick'], topK: readonly LogitCandidate[]): number {
  return pick && pick.topK === topK && pick.index < topK.length ? pick.index : 0
}

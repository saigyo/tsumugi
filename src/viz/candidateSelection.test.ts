import { expect, test } from 'vitest'
import { resetCandidateSelection, selectedCandidate, useCandidateSelection } from './candidateSelection'

test('selectedCandidate: the pick on its own cycle, else the top candidate', () => {
  expect(selectedCandidate(null, 0, 3)).toBe(0)
  expect(selectedCandidate({ cycle: 1, index: 2 }, 1, 3)).toBe(2)
  expect(selectedCandidate({ cycle: 1, index: 2 }, 0, 3)).toBe(0)
})

test('selectedCandidate: an index past this cycle’s candidates falls back to 0', () => {
  expect(selectedCandidate({ cycle: 0, index: 5 }, 0, 3)).toBe(0)
})

test('select stores the pick; reset clears it', () => {
  useCandidateSelection.getState().select(2, 1)
  expect(useCandidateSelection.getState().pick).toEqual({ cycle: 2, index: 1 })
  resetCandidateSelection()
  expect(useCandidateSelection.getState().pick).toBeNull()
})

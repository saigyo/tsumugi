import { expect, test } from 'vitest'
import type { LogitCandidate } from '../trace/types'
import { resetCandidateSelection, selectedCandidate, useCandidateSelection } from './candidateSelection'

const list = (n: number): LogitCandidate[] => Array.from({ length: n }, (_, i) => ({ id: i, text: `t${i}`, logit: 9 - i }))

test('selectedCandidate: the pick on its own candidate list, else the top candidate', () => {
  const a = list(3), b = list(3)
  expect(selectedCandidate(null, a)).toBe(0)
  expect(selectedCandidate({ topK: a, index: 2 }, a)).toBe(2)
  expect(selectedCandidate({ topK: a, index: 2 }, b)).toBe(0)   // another cycle or run
})

test('selectedCandidate: an index past the list falls back to 0', () => {
  const a = list(3)
  expect(selectedCandidate({ topK: a, index: 5 }, a)).toBe(0)
})

test('select stores the pick; reset clears it', () => {
  const a = list(2)
  useCandidateSelection.getState().select(a, 1)
  expect(useCandidateSelection.getState().pick).toEqual({ topK: a, index: 1 })
  resetCandidateSelection()
  expect(useCandidateSelection.getState().pick).toBeNull()
})

import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { resetCandidateSelection } from '../viz/candidateSelection'

// Unit tests never touch the network: any un-stubbed fetch rejects immediately.
// Tests that need fetch install their own stub with vi.stubGlobal('fetch', …).
vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in unit tests'))))

// the candidate pick is a global store; tests start from the top candidate
afterEach(() => resetCandidateSelection())

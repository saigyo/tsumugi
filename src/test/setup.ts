import '@testing-library/jest-dom/vitest'
import { afterEach, beforeEach, vi } from 'vitest'
import { resetNeuronCache } from '../neurons/asset'
import { resetCandidateSelection } from '../viz/candidateSelection'
import { resetChatFolds } from '../viz/chatFolds'
import { resetMlpLayer } from '../viz/mlpLayerSelection'
import { resetNeuronMode } from '../viz/neuronMode'

// Unit tests never touch the network: any un-stubbed fetch rejects immediately.
// Tests that need fetch install their own stub with vi.stubGlobal('fetch', …).
vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in unit tests'))))

// the candidate pick is a global store; tests start from the top candidate
afterEach(() => {
  resetCandidateSelection()
  resetChatFolds()
  resetMlpLayer()
  resetNeuronMode()
  resetNeuronCache()
})

// Node 26 defines its own global localStorage (undefined without
// --localstorage-file), which hides jsdom's; point it back at jsdom's.
beforeEach(() => {
  const jsdom = (globalThis as { jsdom?: { window: Window } }).jsdom
  if (jsdom) vi.stubGlobal('localStorage', jsdom.window.localStorage)
})

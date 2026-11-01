import { create } from 'zustand'
import type { KnockoutLayer } from '../trace/types'

// The Knockouts card asks the app to measure a cycle or to run with one layer
// off; the app installs the handlers (null until a real model is ready) and
// publishes the loaded model's capabilities and the sweep in flight.
interface KnockoutRequestState {
  model: { knockout: boolean; revision: string } | null
  running: { recordId: string; cycle: number; baseline: number | null; layers: KnockoutLayer[] } | null
  error: { recordId: string; cycle: number; message: string } | null   // the last sweep that failed
  measure: ((cycle: number) => void) | null
  runKnockout: ((layer: number, cycle: number) => void) | null
}
export const useKnockoutRequest = create<KnockoutRequestState>(() => ({
  model: null, running: null, error: null, measure: null, runKnockout: null,
}))

export const resetKnockoutRequest = () => useKnockoutRequest.setState({ model: null, running: null, error: null, measure: null, runKnockout: null })

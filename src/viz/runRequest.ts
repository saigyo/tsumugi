import { create } from 'zustand'

// The run-end detail asks the app to continue a finished run with a larger
// token limit; the app installs the handler (null until a real model is ready),
// publishes the loaded model's revision (null revision = the stock model) and
// whether a run is in flight.
interface RunRequestState {
  model: { revision: string | null } | null
  busy: boolean
  continueRun: ((maxNewTokens: number) => void) | null
}
export const useRunRequest = create<RunRequestState>(() => ({
  model: null, busy: false, continueRun: null,
}))

export const resetRunRequest = () => useRunRequest.setState({ model: null, busy: false, continueRun: null })

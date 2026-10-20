import { create } from 'zustand'

// Which neurons "Inside one MLP" lists. A store rather than component state:
// the card remounts when another run is opened, and the reader's choice
// should stay.
export type NeuronMode = 'vote' | 'size'

export const useNeuronMode = create<{ mode: NeuronMode; setMode: (m: NeuronMode) => void }>((set) => ({
  mode: 'vote',
  setMode: (mode) => set({ mode }),
}))

export const resetNeuronMode = () => useNeuronMode.setState({ mode: 'vote' })

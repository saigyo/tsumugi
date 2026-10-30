import { create } from 'zustand'

// Any card can ask the app to load the real model; the app installs the
// handler and keeps loading up to date.
interface RealModelState { request: () => void; loading: boolean }
export const useRealModel = create<RealModelState>(() => ({ request: () => {}, loading: false }))

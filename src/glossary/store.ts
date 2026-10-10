import { create } from 'zustand'
import type { GlossaryId } from './types'

export type GlossaryView = { kind: 'index' } | { kind: 'entry'; id: GlossaryId; section?: string }
export type FocusTarget = HTMLElement | SVGElement

interface GlossaryState {
  view: GlossaryView | null          // null = drawer closed
  history: GlossaryView[]            // views to go Back to, oldest first
  opener: FocusTarget | null         // gets focus back on close
  openEntry: (id: GlossaryId, section?: string, opener?: FocusTarget) => void
  openIndex: (opener?: FocusTarget) => void
  back: () => void
  close: () => void
}

const sameView = (a: GlossaryView, b: GlossaryView) =>
  a.kind === b.kind && (a.kind === 'index' || (b.kind === 'entry' && a.id === b.id && a.section === b.section))

const initial = { view: null, history: [], opener: null }

export const useGlossaryStore = create<GlossaryState>()((set, get) => {
  const go = (next: GlossaryView, opener?: FocusTarget) => {
    const { view, history } = get()
    if (view === null) set({ view: next, history: [], opener: opener ?? null })
    else if (!sameView(view, next)) set({ view: next, history: [...history, view] })
  }
  return {
    ...initial,
    openEntry: (id, section, opener) => go(section ? { kind: 'entry', id, section } : { kind: 'entry', id }, opener),
    openIndex: (opener) => go({ kind: 'index' }, opener),
    back: () => {
      const { history } = get()
      if (history.length === 0) return
      set({ view: history[history.length - 1], history: history.slice(0, -1) })
    },
    close: () => {
      const { opener } = get()
      set(initial)
      // the opener may have unmounted while the drawer was open (playback moved
      // to another card) — fall back to the "?" button
      const target = opener?.isConnected ? opener
        : document.querySelector<HTMLElement>('[data-testid="btn-help"]')
      target?.focus()
    },
  }
})

export function resetGlossaryStore(): void {
  useGlossaryStore.setState(initial)
}

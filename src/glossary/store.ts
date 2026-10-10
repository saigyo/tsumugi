import { create } from 'zustand'
import type { GlossaryId } from './types'

export type GlossaryView = { kind: 'index' } | { kind: 'entry'; id: GlossaryId; section?: string }
export type FocusTarget = HTMLElement | SVGElement

interface GlossaryState {
  view: GlossaryView | null          // null = drawer closed
  history: GlossaryView[]            // views to go Back to, oldest first
  opener: FocusTarget | null         // gets focus back on close
  returnedFocus: FocusTarget | null  // the element close() just focused; its focus handler skips the tooltip
  openEntry: (id: GlossaryId, section?: string, opener?: FocusTarget) => void
  openIndex: (opener?: FocusTarget) => void
  back: () => void
  close: (via?: 'keyboard' | 'pointer') => void
}

const sameView = (a: GlossaryView, b: GlossaryView) =>
  a.kind === b.kind && (a.kind === 'index' || (b.kind === 'entry' && a.id === b.id && a.section === b.section))

const initial = { view: null, history: [], opener: null, returnedFocus: null }

export const useGlossaryStore = create<GlossaryState>()((set, get) => {
  const go = (next: GlossaryView, opener?: FocusTarget) => {
    const { view, history } = get()
    if (view === null) { set({ view: next, history: [], opener: opener ?? null }); return }
    if (sameView(view, next)) return
    // a term on the page (not a link inside the drawer) is where focus should go back to
    const fromPage = opener != null && opener.closest('.gl-drawer') === null
    set({ view: next, history: [...history, view], ...(fromPage ? { opener } : {}) })
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
    close: (via = 'keyboard') => {
      const { opener } = get()
      // the opener may have unmounted while the drawer was open (playback moved
      // to another card) — fall back to the "?" button
      const target = opener?.isConnected ? opener
        : document.querySelector<HTMLElement>('[data-testid="btn-help"]')
      set({ ...initial, returnedFocus: target ?? null })
      if (!target) return
      // focus always goes back (keyboard and screen-reader users land where they
      // were); after a mouse close it goes back without the focus outline
      if (via === 'pointer') {
        target.setAttribute('data-quiet-focus', '')
        target.addEventListener('blur', () => target.removeAttribute('data-quiet-focus'), { once: true })
      }
      target.focus()
    },
  }
})

export function resetGlossaryStore(): void {
  useGlossaryStore.setState(initial)
}

// True once for the element close() just returned focus to: focus handlers use
// it to skip showing a tooltip that the user did not ask for.
export function consumeReturnedFocus(el: Element): boolean {
  if (useGlossaryStore.getState().returnedFocus !== el) return false
  useGlossaryStore.setState({ returnedFocus: null })
  return true
}

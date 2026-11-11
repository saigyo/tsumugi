import { useEffect, useState, type ReactNode } from 'react'
import { useGlossaryStore } from './store'
import { Icon } from '../ui/Icon'

const HINT_KEY = 'tsumugi.glossaryHintDismissed'

function readDismissed(): boolean {
  try { return localStorage.getItem(HINT_KEY) === '1' } catch { return false }
}
function writeDismissed(): void {
  try { localStorage.setItem(HINT_KEY, '1') } catch { /* private window: the hint returns next visit */ }
}

// The header's "?" (opens the glossary index) and, on a first visit, a
// one-time pointer to "How to read Tsumugi".
export function HelpButton({ tools }: { tools?: ReactNode }) {
  const [hint, setHint] = useState(() => !readDismissed())
  const { openIndex, openEntry } = useGlossaryStore.getState()
  const dismiss = () => { setHint(false); writeDismissed() }

  // opening anything in the glossary counts as having found it
  useEffect(() => useGlossaryStore.subscribe((s, prev) => {
    if (s.view !== null && prev.view === null) dismiss()
  }), [])

  return (
    <div className="gl-help">
      {hint && (
        <span data-testid="glossary-hint" className="gl-hint">
          New here? Start with{' '}
          <button type="button" data-testid="btn-hint-open" className="gl-link"
            onClick={(e) => openEntry('how-to-read', undefined, e.currentTarget)}>How to read Tsumugi</button>
          <button type="button" data-testid="btn-hint-dismiss" className="btn btn-quiet btn-icon btn-sm"
            aria-label="Dismiss hint" onClick={dismiss}><Icon name="close" /></button>
        </span>
      )}
      {tools}
      <button type="button" data-testid="btn-help" className="btn btn-icon" aria-label="Open the glossary"
        onClick={(e) => openIndex(e.currentTarget)}>?</button>
    </div>
  )
}

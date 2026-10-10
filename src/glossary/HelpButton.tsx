import { useEffect, useState } from 'react'
import { useGlossaryStore } from './store'

const HINT_KEY = 'tsumugi.glossaryHintDismissed'

function readDismissed(): boolean {
  try { return localStorage.getItem(HINT_KEY) === '1' } catch { return false }
}
function writeDismissed(): void {
  try { localStorage.setItem(HINT_KEY, '1') } catch { /* private window: the hint returns next visit */ }
}

// The header's "?" (opens the glossary index) and, on a first visit, a
// one-time pointer to "How to read Tsumugi".
export function HelpButton() {
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
          <button type="button" data-testid="btn-hint-dismiss" className="gl-hint-dismiss"
            aria-label="Dismiss hint" onClick={dismiss}>×</button>
        </span>
      )}
      <button type="button" data-testid="btn-help" className="gl-help-btn" aria-label="Open the glossary"
        onClick={(e) => openIndex(e.currentTarget)}>?</button>
    </div>
  )
}

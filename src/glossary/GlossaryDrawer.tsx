import { Component, useEffect, useRef, type ReactNode } from 'react'
import { getEntry, indexGroups, STAGE_LABEL } from './registry'
import { useGlossaryStore, type GlossaryView } from './store'
import type { GlossaryEntry, RunSummary } from './types'

const FLASH_MS = 1500

// An entry's "In this run" block must never take the entry down with it.
class InRunBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? null : this.props.children }
}

function InRunBlock({ entry, run }: { entry: GlossaryEntry; run: RunSummary }) {
  const content = entry.inRun?.(run)
  if (content == null || content === false) return null
  return (
    <div data-testid="gl-in-run" className="gl-in-run">
      <div className="gl-in-run-label">In this run</div>
      {content}
    </div>
  )
}

const viewKey = (v: GlossaryView | null) =>
  v === null ? '' : v.kind === 'index' ? 'index' : `${v.id}#${v.section ?? ''}`

// The glossary panel: fixed to the right edge, non-modal, one view at a time
// (an entry or the index). Mounted once in App.
export function GlossaryDrawer({ run }: { run: RunSummary | null }) {
  const view = useGlossaryStore((s) => s.view)
  const canGoBack = useGlossaryStore((s) => s.history.length > 0)
  const { openEntry, openIndex, back, close } = useGlossaryStore.getState()
  const headingRef = useRef<HTMLHeadingElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const key = viewKey(view)
  const kind = view?.kind
  const section = view?.kind === 'entry' ? view.section : undefined

  // focus and scroll only when the view changes (key also covers switching
  // between entries) — never on a run re-render
  useEffect(() => {
    if (!kind) return
    headingRef.current?.focus()
    if (!section) { bodyRef.current?.scrollTo?.({ top: 0 }); return }
    const el = [...(bodyRef.current?.querySelectorAll('[data-gl-section]') ?? [])]
      .find((s) => s.getAttribute('data-gl-section') === section)
    if (!el) return
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    el.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' })
    el.classList.add('gl-flash')
    const t = setTimeout(() => el.classList.remove('gl-flash'), FLASH_MS)
    return () => { clearTimeout(t); el.classList.remove('gl-flash') }
  }, [key, kind, section])

  useEffect(() => {
    if (!view) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [view, close])

  if (!view) return null
  const entry = view.kind === 'entry' ? getEntry(view.id) : undefined

  return (
    <aside data-testid="glossary-drawer" className="gl-drawer" aria-label="Glossary">
      <div className="gl-drawer-bar">
        {canGoBack
          ? <button type="button" data-testid="gl-back" className="gl-link" onClick={back}>← Back</button>
          : <span />}
        <button type="button" data-testid="gl-close" className="gl-close" aria-label="Close glossary" onClick={close}>×</button>
      </div>
      <div ref={bodyRef} className="gl-drawer-body">
        {entry ? (
          <>
            <h2 ref={headingRef} tabIndex={-1} data-testid="gl-heading" className="gl-heading">
              {entry.title} <span className="gl-stage-chip">{STAGE_LABEL[entry.stage]}</span>
            </h2>
            <p className="gl-summary">{entry.summary}</p>
            {entry.body()}
            {run && entry.inRun && (
              <InRunBoundary key={entry.id}><InRunBlock entry={entry} run={run} /></InRunBoundary>
            )}
            {entry.related.length > 0 && (
              <div className="gl-related">
                <span className="gl-related-label">Related:</span>
                {entry.related.map((id) => {
                  const r = getEntry(id)
                  return r && (
                    <button key={id} type="button" data-testid="gl-related" className="gl-link"
                      onClick={() => openEntry(id)}>{r.title}</button>
                  )
                })}
              </div>
            )}
            <button type="button" data-testid="gl-all-entries" className="gl-link gl-all" onClick={() => openIndex()}>
              All entries
            </button>
          </>
        ) : (
          <>
            <h2 ref={headingRef} tabIndex={-1} data-testid="gl-heading" className="gl-heading">Glossary</h2>
            <IndexRow id="how-to-read" />
            {indexGroups().map((g) => (
              <section key={g.stage} data-testid="gl-index-group" data-stage={g.stage} className="gl-index-group">
                <h3>{g.label}</h3>
                {g.entries.map((e) => <IndexRow key={e.id} id={e.id} />)}
              </section>
            ))}
          </>
        )}
      </div>
    </aside>
  )
}

function IndexRow({ id }: { id: GlossaryEntry['id'] }) {
  const entry = getEntry(id)
  const openEntry = useGlossaryStore((s) => s.openEntry)
  if (!entry) return null
  return (
    <button type="button" data-testid="gl-index-row" data-entry={id} className="gl-index-row" onClick={() => openEntry(id)}>
      <span className="gl-index-title">{entry.title}</span>
      <span className="gl-index-summary">{entry.summary}</span>
    </button>
  )
}

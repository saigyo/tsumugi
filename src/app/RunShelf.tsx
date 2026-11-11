import { useEffect, useRef, useState } from 'react'
import { DEFAULT_SYSTEM_PROMPT } from '../engine/chatTemplate'
import { ShareLinkButton, type ShareFeedback } from '../viz/ShareLinkButton'
import type { RunRecord } from './runsStore'
import { shareableRun } from './shareLink'
import { shelfRow } from './shelfRow'
import { Icon } from '../ui/Icon'

export interface RunShelfProps {
  records: RunRecord[]
  activeId: string | null
  compare: { aId: string; bId: string } | null
  armed: boolean
  persistFailed: boolean
  importError: string | null
  onActivate(id: string): void
  onSelectCompareB(id: string): void
  onTogglePin(id: string): void
  onExport(id: string): void
  onRemove(id: string): void
}

const isChat = (r: RunRecord) => r.meta.format?.kind === 'chat'
const words = (r: RunRecord) => r.meta.prompt.trim().split(/\s+/).slice(0, 3).join(' ')
const knockoutNote = (r: RunRecord) => (r.meta.knockout ? ` · MLP ${r.meta.knockout.layer} off` : '')
// the resting chip is short; temperature and Chat stay in the title and the full list
const chipLabel = (r: RunRecord) => `#${r.meta.seq} · ${words(r)}${knockoutNote(r)}`
const listLabel = (r: RunRecord) =>
  `#${r.meta.seq} · ${isChat(r) ? 'Chat · ' : ''}${words(r)} · T=${r.meta.params.temperature}${knockoutNote(r)}`
const systemNote = (r: RunRecord): string => {
  const f = r.meta.format
  if (f?.kind !== 'chat') return ''
  if (f.system === null) return ' · system: none'
  if (f.system === DEFAULT_SYSTEM_PROMPT) return ' · system: default'
  return ` · system: ${f.system.length > 60 ? `${f.system.slice(0, 60)}…` : f.system}`
}

export function RunShelf(p: RunShelfProps) {
  const [share, setShare] = useState<{ runId: string; feedback: ShareFeedback } | null>(null)
  const [open, setOpen] = useState(false)
  const wrap = useRef<HTMLDivElement>(null)
  const moreBtn = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!open) return
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') { setOpen(false); moreBtn.current?.focus() } }
    const away = (e: MouseEvent) => { if (!wrap.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('keydown', key)
    document.addEventListener('mousedown', away)
    return () => { document.removeEventListener('keydown', key); document.removeEventListener('mousedown', away) }
  }, [open])
  const rowEl = useRef<HTMLDivElement>(null)
  useEffect(() => {
    // keep the active chip visible when the row scrolls sideways; only the row scrolls, never the window
    const row = rowEl.current
    const chip = row?.querySelector<HTMLElement>('[data-active="true"]')
    if (!row || !chip) return
    // offsetLeft is measured from the nearest positioned ancestor (the row is not one), so use bounding rects
    const rowBox = row.getBoundingClientRect()
    const chipBox = chip.getBoundingClientRect()
    const left = chipBox.left - rowBox.left + row.scrollLeft
    const right = left + chipBox.width
    if (left < row.scrollLeft) row.scrollLeft = left
    else if (right > row.scrollLeft + row.clientWidth) row.scrollLeft = right - row.clientWidth
  }, [p.activeId])
  const hiddenCount = shelfRow(p.records, [p.activeId, p.compare?.aId, p.compare?.bId]).hidden
  // no hidden runs left: forget that the list was open, so it stays closed when a later run hides one again
  if (hiddenCount === 0 && open) setOpen(false)
  // after the list closes its row unmounts: focus returns to the button that opened it, or to the chosen chip
  const refocus = useRef<string | null>(null)
  useEffect(() => {
    const id = refocus.current
    if (id === null) return
    refocus.current = null
    if (moreBtn.current) moreBtn.current.focus()
    else rowEl.current?.querySelector<HTMLElement>(`[data-run-id="${id}"] [data-testid="run-chip-main"]`)?.focus()
  })
  const notes = (
    <>
      {p.importError && <span data-testid="import-error" className="error"><Icon name="info" />{p.importError}</span>}
      {p.persistFailed && <span data-testid="shelf-note" className="fine">Archive not persisted in this browser.</span>}
    </>
  )
  if (p.records.length === 0) {
    return (p.importError || p.persistFailed) ? <div data-testid="run-shelf" className="run-shelf">{notes}</div> : null
  }
  const selecting = p.armed || p.compare !== null
  const aId = p.compare?.aId ?? (p.armed ? p.activeId : null)
  const chipClick = (id: string) => {
    if (selecting) { if (id !== aId) p.onSelectCompareB(id); return }
    p.onActivate(id)
  }
  // feedback belongs to the run it came from: it disappears with that run
  const shown = share && p.records.some((r) => r.id === share.runId) ? share.feedback : null
  const role = (id: string) => (id === aId ? 'a' : id === p.compare?.bId ? 'b' : '')
  // an A or B run is picked for the comparison; otherwise the active run is the view
  const stateClass = (id: string) => role(id) ? ' is-picked' : id === p.activeId ? ' is-view' : ''
  const { row, hidden } = shelfRow(p.records, [p.activeId, p.compare?.aId, p.compare?.bId])
  const listOpen = open && hidden > 0
  const actions = (r: RunRecord) => (
    <>
      <button data-testid="btn-chip-pin" className="run-chip-action run-chip-pin"
        data-pinned={String(r.meta.pinned)} aria-pressed={r.meta.pinned}
        aria-label={r.meta.pinned ? 'unpin run' : 'pin run'}
        title={r.meta.pinned ? 'unpin' : 'pin — never evicted'} onClick={() => p.onTogglePin(r.id)}><Icon name={r.meta.pinned ? 'pinned' : 'pin'} /></button>
      {(() => {
        const settings = shareableRun(r.events)
        return settings && (
          <ShareLinkButton compact settings={settings}
            onFeedback={(feedback) => setShare((cur) => feedback
              ? { runId: r.id, feedback }
              : cur?.runId === r.id ? null : cur)} />
        )
      })()}
      <button data-testid="btn-chip-export" className="run-chip-action"
        title="download run as JSON" aria-label="download run as JSON" onClick={() => p.onExport(r.id)}><Icon name="download" /></button>
      <button data-testid="btn-chip-remove" className="run-chip-action"
        title="remove run" aria-label="remove run" onClick={() => p.onRemove(r.id)}><Icon name="close" /></button>
    </>
  )
  return (
    <div data-testid="run-shelf" className="run-shelf">
      <div className="run-line">
        <span className="run-label">Runs</span>
        <div className="run-row" ref={rowEl}>
          {row.map((r) => (
            <div key={r.id} data-testid="run-chip" className={`run-chip${stateClass(r.id)}`}
              data-run-id={r.id}
              data-active={String(r.id === p.activeId)} data-pinned={String(r.meta.pinned)} data-role={role(r.id) || undefined}>
              <button data-testid="run-chip-main" className="run-chip-main" onClick={() => chipClick(r.id)}
                title={`${r.meta.prompt} — ${r.meta.mode} · T=${r.meta.params.temperature}${systemNote(r)} · ended: ${r.meta.reason}`}>
                <span className="run-chip-glyph" aria-hidden="true">{r.meta.mode === 'real' ? '●' : '○'}</span>
                {chipLabel(r)}
              </button>
              {r.meta.pinned && <span className="run-chip-pinmark" aria-hidden="true"><Icon name="pinned" /></span>}
              <span className="run-chip-actions">{actions(r)}</span>
            </div>
          ))}
        </div>
        {hidden > 0 && (
          <div className="run-more-wrap" ref={wrap}>
            <button ref={moreBtn} data-testid="run-more" className="run-more pill" aria-expanded={listOpen}
              onClick={() => setOpen((o) => !o)}>+{hidden} more <Icon name="chevron" /></button>
            {listOpen && (
              <div data-testid="run-list" className="run-list pop">
                {[...p.records].reverse().map((r) => (
                  <div key={r.id} data-testid="run-list-row" className={`run-list-row${stateClass(r.id)}`}
                    data-active={String(r.id === p.activeId)} data-pinned={String(r.meta.pinned)} data-role={role(r.id) || undefined}>
                    <button data-testid="run-list-main" className="run-list-main" onClick={() => { chipClick(r.id); if (!(selecting && r.id === aId)) { setOpen(false); refocus.current = r.id } }}
                      title={r.meta.prompt}>
                      <span className="run-chip-glyph" aria-hidden="true">{r.meta.mode === 'real' ? '●' : '○'}</span>
                      {listLabel(r)}
                    </button>
                    <span className="run-list-actions">{actions(r)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      <div className="run-notes">
        {p.armed && p.compare === null && (
          <span data-testid="compare-hint" className="note">Pick a second run to compare with #{p.records.find((r) => r.id === aId)?.meta.seq}.</span>
        )}
        {shown?.kind === 'copied' && <span role="status" data-testid="share-copied" className="fine">Link copied</span>}
        {shown?.kind === 'failed' && (
          <input readOnly data-testid="share-link-text" className="share-link-text" value={shown.link}
            aria-label="Share link" onFocus={(e) => e.currentTarget.select()} />
        )}
        {notes}
      </div>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { DEFAULT_SYSTEM_PROMPT } from '../engine/chatTemplate'
import { ShareLinkButton, type ShareFeedback } from '../viz/ShareLinkButton'
import type { RunRecord } from './runsStore'
import { shareableRun } from './shareLink'
import { shelfRow } from './shelfRow'

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
    // keep the active chip visible when the row scrolls sideways
    rowEl.current?.querySelector('[data-active="true"]')?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' })
  }, [p.activeId])
  const notes = (
    <>
      {p.importError && <span data-testid="import-error" className="run-shelf-note">{p.importError}</span>}
      {p.persistFailed && <span data-testid="shelf-note" className="run-shelf-note">archive not persisted in this browser</span>}
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
  const { row, hidden } = shelfRow(p.records, [p.activeId, p.compare?.aId, p.compare?.bId])
  const listOpen = open && hidden > 0
  const actions = (r: RunRecord) => (
    <>
      <button data-testid="btn-chip-pin" className="run-chip-action run-chip-pin"
        data-pinned={String(r.meta.pinned)} aria-pressed={r.meta.pinned}
        aria-label={r.meta.pinned ? 'unpin run' : 'pin run'}
        title={r.meta.pinned ? 'unpin' : 'pin — never evicted'} onClick={() => p.onTogglePin(r.id)}>📌</button>
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
        title="download run as JSON" onClick={() => p.onExport(r.id)}>⇩</button>
      <button data-testid="btn-chip-remove" className="run-chip-action"
        title="remove run" onClick={() => p.onRemove(r.id)}>×</button>
    </>
  )
  return (
    <div data-testid="run-shelf" className="run-shelf">
      <div className="run-line">
        <span className="run-label">Runs</span>
        <div className="run-row" ref={rowEl}>
          {row.map((r) => (
            <div key={r.id} data-testid="run-chip" className="run-chip"
              data-active={String(r.id === p.activeId)} data-pinned={String(r.meta.pinned)} data-role={role(r.id) || undefined}>
              <button data-testid="run-chip-main" className="run-chip-main" onClick={() => chipClick(r.id)}
                title={`${r.meta.prompt} — ${r.meta.mode} · T=${r.meta.params.temperature}${systemNote(r)} · ended: ${r.meta.reason}`}>
                <span className="run-chip-glyph" aria-hidden="true">{r.meta.mode === 'real' ? '●' : '○'}</span>
                {chipLabel(r)}
              </button>
              {r.meta.pinned && <span className="run-chip-pinmark" aria-hidden="true">📌</span>}
              <span className="run-chip-actions">{actions(r)}</span>
            </div>
          ))}
        </div>
        {hidden > 0 && (
          <div className="run-more-wrap" ref={wrap}>
            <button ref={moreBtn} data-testid="run-more" className="run-more" aria-expanded={listOpen}
              onClick={() => setOpen((o) => !o)}>+{hidden} more ▾</button>
            {listOpen && (
              <div data-testid="run-list" className="run-list">
                {[...p.records].reverse().map((r) => (
                  <div key={r.id} data-testid="run-list-row" className="run-list-row"
                    data-active={String(r.id === p.activeId)} data-pinned={String(r.meta.pinned)} data-role={role(r.id) || undefined}>
                    <button data-testid="run-list-main" className="run-list-main" onClick={() => { chipClick(r.id); if (!(selecting && r.id === aId)) setOpen(false) }}
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
        {shown?.kind === 'copied' && <span role="status" data-testid="share-copied" className="run-shelf-note">Link copied</span>}
        {shown?.kind === 'failed' && (
          <input readOnly data-testid="share-link-text" className="share-link-text" value={shown.link}
            aria-label="Share link" onFocus={(e) => e.currentTarget.select()} />
        )}
        {notes}
      </div>
    </div>
  )
}

import { useState } from 'react'
import { DEFAULT_SYSTEM_PROMPT } from '../engine/chatTemplate'
import { ShareLinkButton, type ShareFeedback } from '../viz/ShareLinkButton'
import type { RunRecord } from './runsStore'
import { shareableRun } from './shareLink'

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
const label = (r: RunRecord) =>
  `#${r.meta.seq} · ${isChat(r) ? 'Chat · ' : ''}${r.meta.prompt.trim().split(/\s+/).slice(0, 3).join(' ')} · T=${r.meta.params.temperature}${r.meta.knockout ? ` · MLP ${r.meta.knockout.layer} off` : ''}`
const systemNote = (r: RunRecord): string => {
  const f = r.meta.format
  if (f?.kind !== 'chat') return ''
  if (f.system === null) return ' · system: none'
  if (f.system === DEFAULT_SYSTEM_PROMPT) return ' · system: default'
  return ` · system: ${f.system.length > 60 ? `${f.system.slice(0, 60)}…` : f.system}`
}

export function RunShelf(p: RunShelfProps) {
  const [share, setShare] = useState<{ runId: string; feedback: ShareFeedback } | null>(null)
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
  return (
    <div data-testid="run-shelf" className="run-shelf">
      {p.records.map((r) => (
        <div key={r.id} data-testid="run-chip" className="run-chip"
          data-active={String(r.id === p.activeId)} data-pinned={String(r.meta.pinned)} data-role={role(r.id)}>
          <button data-testid="run-chip-main" className="run-chip-main" onClick={() => chipClick(r.id)}
            title={`${r.meta.prompt} — ${r.meta.mode}${systemNote(r)} · ended: ${r.meta.reason}`}>
            <span className="run-chip-glyph" aria-hidden="true">{r.meta.mode === 'real' ? '●' : '○'}</span>
            {label(r)}
          </button>
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
        </div>
      ))}
      {shown?.kind === 'copied' && <span role="status" data-testid="share-copied" className="run-shelf-note">Link copied</span>}
      {shown?.kind === 'failed' && (
        <input readOnly data-testid="share-link-text" className="share-link-text" value={shown.link}
          aria-label="Share link" onFocus={(e) => e.currentTarget.select()} />
      )}
      {notes}
    </div>
  )
}

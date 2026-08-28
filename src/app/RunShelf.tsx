import { useRef } from 'react'
import type { RunRecord } from './runsStore'

export interface RunShelfProps {
  records: RunRecord[]
  activeId: string | null
  compare: { aId: string; bId: string } | null
  armed: boolean
  persistFailed: boolean
  importError: string | null
  onActivate(id: string): void
  onSelectCompareB(id: string): void
  onArmCompare(): void
  onExitCompare(): void
  onTogglePin(id: string): void
  onExport(id: string): void
  onRemove(id: string): void
  onImportFile(file: File): void
}

const label = (r: RunRecord) =>
  `#${r.meta.seq} · ${r.meta.prompt.trim().split(/\s+/).slice(0, 3).join(' ')} · T=${r.meta.params.temperature}`

export function RunShelf(p: RunShelfProps) {
  const fileRef = useRef<HTMLInputElement>(null)
  if (p.records.length === 0) return null
  const selecting = p.armed || p.compare !== null
  const aId = p.compare?.aId ?? (p.armed ? p.activeId : null)
  const chipClick = (id: string) => {
    if (selecting) { if (id !== aId) p.onSelectCompareB(id); return }
    p.onActivate(id)
  }
  const role = (id: string) => (id === aId ? 'a' : id === p.compare?.bId ? 'b' : '')
  return (
    <div data-testid="run-shelf" className="run-shelf">
      {p.records.map((r) => (
        <div key={r.id} data-testid="run-chip" className="run-chip"
          data-active={String(r.id === p.activeId)} data-pinned={String(r.meta.pinned)} data-role={role(r.id)}>
          <button data-testid="run-chip-main" className="run-chip-main" onClick={() => chipClick(r.id)}
            title={`${r.meta.prompt} — ${r.meta.mode} · ended: ${r.meta.reason}`}>
            <span className="run-chip-glyph" aria-hidden="true">{r.meta.mode === 'real' ? '●' : '○'}</span>
            {label(r)}{r.meta.pinned && ' 📌'}
          </button>
          <button data-testid="btn-chip-pin" className="run-chip-action"
            title={r.meta.pinned ? 'unpin' : 'pin — never evicted'} onClick={() => p.onTogglePin(r.id)}>📌</button>
          <button data-testid="btn-chip-export" className="run-chip-action"
            title="download run as JSON" onClick={() => p.onExport(r.id)}>⇩</button>
          <button data-testid="btn-chip-remove" className="run-chip-action"
            title="remove run" onClick={() => p.onRemove(r.id)}>×</button>
        </div>
      ))}
      <span className="run-shelf-tools">
        {selecting
          ? <button data-testid="btn-compare-exit" className="run-shelf-btn" onClick={p.onExitCompare}>× exit compare</button>
          : p.records.length > 1 && <button data-testid="btn-compare-arm" className="run-shelf-btn" disabled={p.activeId === null}
              title="pick a second run to compare against the active one" onClick={p.onArmCompare}>⇄ compare</button>}
        <button data-testid="btn-import" className="run-shelf-btn" title="load an exported run"
          onClick={() => fileRef.current?.click()}>⇧ import</button>
        <input ref={fileRef} data-testid="import-input" type="file" accept="application/json" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) p.onImportFile(f); e.target.value = '' }} />
      </span>
      {p.importError && <span data-testid="import-error" className="run-shelf-note">{p.importError}</span>}
      {p.persistFailed && <span data-testid="shelf-note" className="run-shelf-note">archive not persisted in this browser</span>}
    </div>
  )
}

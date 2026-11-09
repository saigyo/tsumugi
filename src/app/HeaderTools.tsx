import { useRef } from 'react'

// Compare and Import live in the app header, left of the help button.
export function HeaderTools(p: {
  runCount: number; canArm: boolean; selecting: boolean
  onArmCompare(): void; onExitCompare(): void; onImportFile(file: File): void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  return (
    <>
      {p.selecting
        ? <button type="button" data-testid="btn-compare-exit" className="hdr-btn" onClick={p.onExitCompare}>× Exit compare</button>
        : p.runCount > 1 && <button type="button" data-testid="btn-compare-arm" className="hdr-btn" disabled={!p.canArm}
            title="pick a second run to compare against the active one" onClick={p.onArmCompare}>⇄ Compare</button>}
      <button type="button" data-testid="btn-import" className="hdr-btn" title="load an exported run"
        onClick={() => fileRef.current?.click()}>⇧ Import</button>
      <input ref={fileRef} data-testid="import-input" type="file" accept="application/json" hidden
        onChange={(e) => { const f = e.target.files?.[0]; if (f) p.onImportFile(f); e.target.value = '' }} />
    </>
  )
}

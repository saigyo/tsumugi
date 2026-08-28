import type { TraceEvent } from '../trace/types'
import type { RunStorage } from './runStorage'
import { useRunsStore, type RunRecord, type SealMeta } from './runsStore'

// Store ↔ storage glue. The store stays pure; this module mirrors every archive
// mutation to the adapter, fire-and-forget. Any storage failure flips the
// store's persistFailed flag — the archive continues session-only, and
// generation is never blocked or delayed (M1 never-fail policy).
let storage: RunStorage | null = null

const markFailed = () => useRunsStore.getState().setPersistFailed()
const mirror = (op: (s: RunStorage) => Promise<unknown>) => {
  if (storage) void op(storage).catch(markFailed)
}

export async function initArchive(s: RunStorage): Promise<void> {
  storage = s
  try {
    useRunsStore.getState().hydrate(await s.loadAll())
  } catch {
    markFailed()
  }
}

export function archiveSeal(meta: SealMeta, events: TraceEvent[]): RunRecord {
  const { record, evicted } = useRunsStore.getState().seal(meta, events)
  mirror((s) => s.put(record))
  for (const r of evicted) mirror((s) => s.delete(r.id))
  return record
}

export function archiveTogglePin(id: string): void {
  useRunsStore.getState().togglePin(id)
  const record = useRunsStore.getState().records.find((r) => r.id === id)
  if (record) mirror((s) => s.put(record))
}

export function archiveRemove(id: string): void {
  useRunsStore.getState().remove(id)
  mirror((s) => s.delete(id))
}

export function archiveImport(data: { meta: SealMeta; events: TraceEvent[] }): RunRecord {
  const record = useRunsStore.getState().importRecord(data)
  mirror((s) => s.put(record))
  return record
}

export function _resetArchiveForTests(): void { storage = null }

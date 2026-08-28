import { create } from 'zustand'
import type { GenParams, Mode, RunEndReason, TraceEvent } from '../trace/types'

export interface RunMeta {
  seq: number               // monotonic per-archive counter; never reused, so chip labels
                            // stay stable across evictions
  prompt: string
  params: GenParams
  mode: Mode
  modelId?: string          // real mode only
  endedAt: number           // epoch ms, stamped at seal time
  reason: RunEndReason
  pinned: boolean
}
export interface RunRecord { id: string; meta: RunMeta; events: TraceEvent[] }
export type SealMeta = Omit<RunMeta, 'seq' | 'pinned'>

export const UNPINNED_CAP = 8

interface RunsState {
  records: RunRecord[]      // oldest first
  activeId: string | null
  nextSeq: number
  persistFailed: boolean
  seal: (meta: SealMeta, events: TraceEvent[]) => { record: RunRecord; evicted: RunRecord[] }
  setActive: (id: string) => void
  togglePin: (id: string) => void
  remove: (id: string) => void
  importRecord: (data: { meta: SealMeta; events: TraceEvent[] }) => RunRecord
  hydrate: (records: RunRecord[]) => void
  setPersistFailed: () => void
}

export const useRunsStore = create<RunsState>()((set, get) => ({
  records: [], activeId: null, nextSeq: 1, persistFailed: false,
  seal: (meta, events) => {
    const record: RunRecord = {
      id: crypto.randomUUID(), meta: { ...meta, seq: get().nextSeq, pinned: false }, events,
    }
    const records = [...get().records, record]
    const evicted: RunRecord[] = []
    while (records.filter((r) => !r.meta.pinned).length > UNPINNED_CAP) {
      const oldest = records.find((r) => !r.meta.pinned)
      if (!oldest) break
      evicted.push(oldest)
      records.splice(records.indexOf(oldest), 1)
    }
    set((s) => ({ records, activeId: record.id, nextSeq: s.nextSeq + 1 }))
    return { record, evicted }
  },
  setActive: (id) => set((s) => (s.records.some((r) => r.id === id) ? { activeId: id } : {})),
  togglePin: (id) => set((s) => ({
    records: s.records.map((r) => r.id === id ? { ...r, meta: { ...r.meta, pinned: !r.meta.pinned } } : r),
  })),
  remove: (id) => set((s) => ({
    records: s.records.filter((r) => r.id !== id),
    activeId: s.activeId === id ? null : s.activeId,
  })),
  importRecord: (data) => {
    const record: RunRecord = {
      id: crypto.randomUUID(),
      meta: { ...data.meta, seq: get().nextSeq, pinned: true },   // pinned: imports must not fall off the ring
      events: data.events,
    }
    set((s) => ({ records: [...s.records, record], nextSeq: s.nextSeq + 1 }))
    return record
  },
  hydrate: (records) => {
    const sorted = [...records].sort((a, b) => a.meta.endedAt - b.meta.endedAt)
    set({ records: sorted, nextSeq: sorted.reduce((m, r) => Math.max(m, r.meta.seq), 0) + 1 })
  },
  setPersistFailed: () => set({ persistFailed: true }),
}))

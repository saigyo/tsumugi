import type { RunRecord } from './runsStore'

export interface RunStorage {
  loadAll(): Promise<RunRecord[]>
  put(record: RunRecord): Promise<void>
  delete(id: string): Promise<void>
}

export function createMemoryStorage(opts: { failing?: boolean } = {}): RunStorage & { map: Map<string, RunRecord> } {
  const map = new Map<string, RunRecord>()
  const fail = () => Promise.reject(new Error('storage unavailable'))
  return {
    map,
    loadAll: () => (opts.failing ? fail() : Promise.resolve([...map.values()])),
    put: (record) => { if (opts.failing) return fail(); map.set(record.id, record); return Promise.resolve() },
    delete: (id) => { if (opts.failing) return fail(); map.delete(id); return Promise.resolve() },
  }
}

// IndexedDB-backed storage. Every rejection is treated by callers as non-fatal
// (session-only archive) — including environments without indexedDB (jsdom, old browsers).
export function createIndexedDbStorage(): RunStorage {
  const open = (): Promise<IDBDatabase> => new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') { reject(new Error('indexedDB unavailable')); return }
    const req = indexedDB.open('tsumugi', 1)
    req.onupgradeneeded = () => { req.result.createObjectStore('runs', { keyPath: 'id' }) }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error ?? new Error('indexedDB open failed'))
  })
  const inTx = async <T,>(mode: IDBTransactionMode, op: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> => {
    const db = await open()
    try {
      return await new Promise<T>((resolve, reject) => {
        const request = op(db.transaction('runs', mode).objectStore('runs'))
        request.onsuccess = () => resolve(request.result)
        request.onerror = () => reject(request.error ?? new Error('indexedDB request failed'))
      })
    } finally { db.close() }
  }
  return {
    loadAll: () => inTx('readonly', (s) => s.getAll() as IDBRequest<RunRecord[]>),
    put: async (record) => { await inTx('readwrite', (s) => s.put(record)) },
    delete: async (id) => { await inTx('readwrite', (s) => s.delete(id)) },
  }
}

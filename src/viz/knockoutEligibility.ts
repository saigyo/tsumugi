import type { TraceEvent } from '../trace/types'

export type Ineligible = 'sim' | 'knockout-run' | 'old-run' | 'no-model' | 'too-long'

const MIN_REVISION = 7
const MAX_SEQ_LEN = 512

function revisionNumber(revision: string): number {
  return /^v\d+$/.test(revision) ? Number(revision.slice(1)) : -1
}

export function knockoutEligibility(
  run: Extract<TraceEvent, { type: 'run-start' }>,
  seqLen: number,
  model: { knockout: boolean; revision: string } | null,
): { ok: true } | { ok: false; reason: Ineligible } {
  const no = (reason: Ineligible) => ({ ok: false as const, reason })
  if (run.mode === 'sim') return no('sim')
  if (run.knockout) return no('knockout-run')
  if (run.seed === undefined || run.revision === undefined || revisionNumber(run.revision) < MIN_REVISION) return no('old-run')
  if (!model || !model.knockout) return no('no-model')
  if (run.revision !== model.revision) return no('old-run')
  if (seqLen > MAX_SEQ_LEN) return no('too-long')
  return { ok: true }
}

import type { GenParams, PromptFormat, RunEndReason, TraceEvent } from '../trace/types'
import { dropBrokenNeurons, validateTrace } from '../trace/validate'
import type { RunMeta, RunRecord, SealMeta } from './runsStore'

export const RUN_FILE_VERSION = 1

export function serializeRun(record: RunRecord): { filename: string; json: string } {
  const slug = record.meta.prompt.trim().toLowerCase().split(/\s+/).slice(0, 3).join('-')
    .replace(/[^a-z0-9-]/g, '') || 'run'
  const date = new Date(record.meta.endedAt).toISOString().slice(0, 10).replace(/-/g, '')
  return {
    filename: `tsumugi-run-${slug}-${date}.json`,
    json: JSON.stringify({ formatVersion: RUN_FILE_VERSION, meta: record.meta, events: record.events }, null, 2),
  }
}

const formatOf = (f: unknown): PromptFormat | undefined => {
  if (typeof f !== 'object' || f === null) return undefined
  const v = f as { kind?: unknown; system?: unknown }
  if (v.kind === 'text') return { kind: 'text' }
  if (v.kind === 'chat' && (v.system === null || typeof v.system === 'string')) return { kind: 'chat', system: v.system }
  return undefined
}

export type ParsedRunFile =
  | { ok: true; meta: SealMeta; events: TraceEvent[]; note?: string }
  | { ok: false; error: string }

export function parseRunFile(text: string): ParsedRunFile {
  let data: unknown
  try { data = JSON.parse(text) } catch { return { ok: false, error: 'not valid JSON' } }
  if (typeof data !== 'object' || data === null) return { ok: false, error: 'not a run file' }
  const d = data as { formatVersion?: unknown; meta?: unknown; events?: unknown }
  if (d.formatVersion !== RUN_FILE_VERSION)
    return { ok: false, error: `unsupported format version (expected ${RUN_FILE_VERSION})` }
  const m = (typeof d.meta === 'object' && d.meta !== null ? d.meta : {}) as Partial<RunMeta>
  if (typeof m.prompt !== 'string' || typeof m.endedAt !== 'number'
    || (m.mode !== 'sim' && m.mode !== 'real') || typeof m.reason !== 'string'
    || typeof m.params !== 'object' || m.params === null)
    return { ok: false, error: 'missing or malformed run metadata' }
  if (!Array.isArray(d.events)) return { ok: false, error: 'missing events' }
  const nonObject = d.events.findIndex((e) => typeof e !== 'object' || e === null)
  if (nonObject >= 0) return { ok: false, error: `invalid trace: event ${nonObject} is not an object` }
  const { events, droppedNeurons, droppedSteps } = dropBrokenNeurons(d.events as TraceEvent[])
  const problems = validateTrace(events)
  if (problems.length > 0) return { ok: false, error: `invalid trace: ${problems[0]}` }
  const damaged = (what: string, n: number) => n > 0 ? [`${what} of ${n} cycle${n === 1 ? '' : 's'} was damaged and left out.`] : []
  const note = [...damaged('Neuron data', droppedNeurons), ...damaged('Step-by-step data', droppedSteps)].join(' ') || undefined
  return {
    ok: true,
    meta: {
      prompt: m.prompt, params: m.params as GenParams, mode: m.mode,
      ...(typeof m.modelId === 'string' ? { modelId: m.modelId } : {}),
      ...(formatOf(m.format) ? { format: formatOf(m.format) } : {}),
      endedAt: m.endedAt, reason: m.reason as RunEndReason,
    },
    events,
    ...(note ? { note } : {}),
  }
}

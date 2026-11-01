import type { GenParams, KnockoutOrigin, KnockoutSweep, PromptFormat, RunEndReason, TraceEvent } from '../trace/types'
import { TEXT_FORMAT } from '../engine/chatTemplate'
import { dropBrokenChat, dropBrokenNeurons, sweepProblems, validateTrace } from '../trace/validate'
import type { RunMeta, RunRecord, SealMeta } from './runsStore'

export const RUN_FILE_VERSION = 1

export function serializeRun(record: RunRecord): { filename: string; json: string } {
  const slug = record.meta.prompt.trim().toLowerCase().split(/\s+/).slice(0, 3).join('-')
    .replace(/[^a-z0-9-]/g, '') || 'run'
  const date = new Date(record.meta.endedAt).toISOString().slice(0, 10).replace(/-/g, '')
  return {
    filename: `tsumugi-run-${slug}-${date}.json`,
    json: JSON.stringify({ formatVersion: RUN_FILE_VERSION, meta: record.meta, events: record.events, ...(record.sweeps?.length ? { sweeps: record.sweeps } : {}) }, null, 2),
  }
}

const formatOf = (f: unknown): PromptFormat | undefined => {
  if (typeof f !== 'object' || f === null) return undefined
  const v = f as { kind?: unknown; system?: unknown }
  if (v.kind === 'text') return TEXT_FORMAT
  if (v.kind === 'chat' && (v.system === null || typeof v.system === 'string')) return { kind: 'chat', system: v.system }
  return undefined
}

const knockoutOf = (k: unknown): KnockoutOrigin | undefined => {
  if (typeof k !== 'object' || k === null) return undefined
  const v = k as Partial<Record<keyof KnockoutOrigin, unknown>>
  return Number.isInteger(v.layer) && Number.isInteger(v.cycle) && typeof v.sourceId === 'string'
    ? { layer: v.layer as number, cycle: v.cycle as number, sourceId: v.sourceId } : undefined
}

export type ParsedRunFile =
  | { ok: true; meta: SealMeta; events: TraceEvent[]; sweeps?: KnockoutSweep[]; note?: string }
  | { ok: false; error: string }

export function parseRunFile(text: string): ParsedRunFile {
  let data: unknown
  try { data = JSON.parse(text) } catch { return { ok: false, error: 'not valid JSON' } }
  if (typeof data !== 'object' || data === null) return { ok: false, error: 'not a run file' }
  const d = data as { formatVersion?: unknown; meta?: unknown; events?: unknown; sweeps?: unknown }
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
  const { events: neuronChecked, droppedNeurons, droppedSteps } = dropBrokenNeurons(d.events as TraceEvent[])
  const { events, droppedChats } = dropBrokenChat(neuronChecked)
  const problems = validateTrace(events)
  const format = formatOf(m.format)
  const layerTotal = events.find((e) => e.type === 'layer')
  const layers = layerTotal && layerTotal.type === 'layer' ? layerTotal.total : 0
  const rawSweeps = Array.isArray(d.sweeps) ? d.sweeps : []
  const sweeps = rawSweeps.filter((s) => sweepProblems(s, layers).length === 0) as KnockoutSweep[]
  const droppedSweeps = rawSweeps.length - sweeps.length
  const knockout = knockoutOf(m.knockout)
  if (problems.length > 0) return { ok: false, error: `invalid trace: ${problems[0]}` }
  const damaged = (what: string, n: number) => n > 0 ? [`${what} of ${n} cycle${n === 1 ? '' : 's'} was damaged and left out.`] : []
  const note = [...damaged('Neuron data', droppedNeurons), ...damaged('Step-by-step data', droppedSteps), ...damaged('Chat template data', droppedChats), ...damaged('Knockout data', droppedSweeps)].join(' ') || undefined
  return {
    ok: true,
    meta: {
      prompt: m.prompt, params: m.params as GenParams, mode: m.mode,
      ...(typeof m.modelId === 'string' ? { modelId: m.modelId } : {}),
      ...(format ? { format } : {}),
      ...(knockout ? { knockout } : {}),
      endedAt: m.endedAt, reason: m.reason as RunEndReason,
    },
    events,
    ...(sweeps.length ? { sweeps } : {}),
    ...(note ? { note } : {}),
  }
}

import type { GenParams, PromptFormat, TraceEvent } from '../trace/types'
import { TEXT_FORMAT } from '../engine/chatTemplate'

export interface ShareSettings { prompt: string; format: PromptFormat; params: GenParams; seed: number; rev?: string }
export interface SharedRun { prompt?: string; format?: PromptFormat; params?: Partial<GenParams>; seed?: number; rev?: string }

const MAX_SEED = 4294967295
const MAX_TOKENS = 100

export function buildShareLink(base: string, s: ShareSettings): string {
  const q = new URLSearchParams({ run: '1', p: s.prompt, f: s.format.kind })
  if (s.format.kind === 'chat' && s.format.system !== null) q.set('s', s.format.system)
  q.set('t', String(s.params.temperature))
  q.set('k', String(s.params.topK))
  q.set('n', String(s.params.maxNewTokens))
  q.set('seed', String(s.seed))
  if (s.rev) q.set('rev', s.rev)
  return `${base}#${q.toString()}`
}

const numberIn = (raw: string | null, min: number, max: number, integer: boolean): number | undefined => {
  if (raw === null || raw.trim() === '') return undefined
  const n = Number(raw)
  return Number.isFinite(n) && n >= min && n <= max && (!integer || Number.isInteger(n)) ? n : undefined
}

export function parseShareLink(hash: string): SharedRun | null {
  const q = new URLSearchParams(hash.replace(/^#/, ''))
  if (q.get('run') !== '1') return null
  const out: SharedRun = {}
  const prompt = q.get('p')
  if (prompt !== null) out.prompt = prompt
  const f = q.get('f')
  if (f === 'text') out.format = TEXT_FORMAT
  else if (f === 'chat') out.format = { kind: 'chat', system: q.get('s') }
  const params: Partial<GenParams> = {}
  const temperature = numberIn(q.get('t'), 0, 2, false)
  const topK = numberIn(q.get('k'), 1, 10, true)
  const maxNewTokens = numberIn(q.get('n'), 1, MAX_TOKENS, true)
  if (temperature !== undefined) params.temperature = temperature
  if (topK !== undefined) params.topK = topK
  if (maxNewTokens !== undefined) params.maxNewTokens = maxNewTokens
  if (Object.keys(params).length) out.params = params
  const seed = numberIn(q.get('seed'), 0, MAX_SEED, true)
  if (seed !== undefined) out.seed = seed
  const rev = q.get('rev')
  if (rev) out.rev = rev
  return out
}

export function shareableRun(events: TraceEvent[]): ShareSettings | null {
  const start = events.find((e) => e.type === 'run-start')
  const end = events.find((e) => e.type === 'run-end')
  if (!start || start.type !== 'run-start' || !end || end.type !== 'run-end') return null
  if (start.mode !== 'real' || start.seed === undefined || start.knockout) return null
  if (end.reason !== 'eos' && end.reason !== 'max-tokens') return null
  return {
    prompt: start.prompt, format: start.format ?? TEXT_FORMAT, params: start.params, seed: start.seed,
    ...(start.revision ? { rev: start.revision } : {}),
  }
}

export const continueLimit = (maxNewTokens: number): number | null =>
  maxNewTokens >= MAX_TOKENS ? null : Math.min(MAX_TOKENS, 2 * maxNewTokens)

// null = blank (random seed), undefined = invalid (keep the previous value)
export function parseSeed(raw: string): number | null | undefined {
  if (raw.trim() === '') return null
  return numberIn(raw, 0, MAX_SEED, true)
}

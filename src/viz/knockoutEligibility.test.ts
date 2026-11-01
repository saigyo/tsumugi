import { describe, expect, it } from 'vitest'
import type { TraceEvent } from '../trace/types'
import { knockoutEligibility } from './knockoutEligibility'

type Run = Extract<TraceEvent, { type: 'run-start' }>
const run = (o: Partial<Run> = {}): Run =>
  ({ type: 'run-start', prompt: 'p', mode: 'real', modelId: 'm', params: {}, seed: 1, revision: 'v7', ...o }) as Run
const model = { knockout: true, revision: 'v7' }

describe('knockoutEligibility', () => {
  it('ok', () => expect(knockoutEligibility(run(), 10, model)).toEqual({ ok: true }))
  it('sim', () => expect(knockoutEligibility(run({ mode: 'sim' }), 10, model)).toEqual({ ok: false, reason: 'sim' }))
  it('knockout-run', () =>
    expect(knockoutEligibility(run({ knockout: { layer: 1, cycle: 0, sourceId: 'x' } }), 10, model)).toEqual({ ok: false, reason: 'knockout-run' }))
  it('old-run: no seed, no revision, old or odd revision', () => {
    for (const o of [{ seed: undefined }, { revision: undefined }, { revision: 'v6' }, { revision: 'dev' }])
      expect(knockoutEligibility(run(o), 10, model)).toEqual({ ok: false, reason: 'old-run' })
  })
  it('no-model', () => {
    expect(knockoutEligibility(run(), 10, null)).toEqual({ ok: false, reason: 'no-model' })
    expect(knockoutEligibility(run(), 10, { knockout: false, revision: 'v7' })).toEqual({ ok: false, reason: 'no-model' })
  })
  it('old-run when revision differs from the model', () =>
    expect(knockoutEligibility(run(), 10, { knockout: true, revision: 'v8' })).toEqual({ ok: false, reason: 'old-run' }))
  it('too-long', () => {
    expect(knockoutEligibility(run(), 512, model)).toEqual({ ok: true })
    expect(knockoutEligibility(run(), 513, model)).toEqual({ ok: false, reason: 'too-long' })
  })
})

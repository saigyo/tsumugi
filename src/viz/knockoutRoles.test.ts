import { describe, expect, it } from 'vitest'
import type { KnockoutSweep } from '../trace/types'
import { knockoutRoles, MAX_ROLE_CHIPS } from './knockoutRoles'

const tok = { id: 1, text: 'a' } as unknown as KnockoutSweep['chosen']
// deltas: per-layer change in probability from a 0.5 baseline
function sweep(deltas: number[], flips: number[] = []): KnockoutSweep {
  return {
    cycle: 0, revision: 'v7', chosen: tok, baseline: 0.5,
    layers: deltas.map((d, layer) => ({ layer, prob: 0.5 + d, top: { ...tok, prob: 0.5 }, flips: flips.includes(layer) })),
  }
}
const flat = (n = 30) => Array<number>(n).fill(0)
const votes = (n = 30, v = 1) => { const a = Array<number>(n).fill(v); return a }

describe('knockoutRoles', () => {
  it('flips -> decisive', () => {
    const d = flat(); d[3] = -0.5
    expect(knockoutRoles(sweep(d, [3]), votes(30, 0))).toEqual([{ layer: 3, role: 'decisive' }])
  })
  it('small vote and a 0.4 drop -> preparatory', () => {
    const d = flat(); d[4] = -0.4
    const v = votes(30, 0); v[0] = 4; v[4] = 0.5
    expect(knockoutRoles(sweep(d), v)).toContainEqual({ layer: 4, role: 'preparatory' })
  })
  it('largest vote and tiny change -> compensated', () => {
    const d = flat(); d[7] = 0.01
    const v = votes(30, 0); v[7] = 5
    expect(knockoutRoles(sweep(d), v)).toEqual([{ layer: 7, role: 'compensated' }])
  })
  it('rise of 0.15 -> dampening, rise of 0.08 -> nothing', () => {
    const d = flat(); d[2] = 0.15; d[3] = 0.08
    const v = votes(30, 0); v[9] = 5
    expect(knockoutRoles(sweep(d), v)).toEqual([{ layer: 2, role: 'dampening' }, { layer: 9, role: 'compensated' }])
  })
  it('a vote at 60% of the largest with no change is compensated, at 40% it is not', () => {
    const d = flat()
    const v = votes(30, 0); v[1] = 5; v[7] = 3; v[8] = 2
    expect(knockoutRoles(sweep(d), v)).toEqual([{ layer: 1, role: 'compensated' }, { layer: 7, role: 'compensated' }])
  })
  it('a 0.12 drop with a small vote is preparatory, with a large vote it is not', () => {
    const d = flat(); d[4] = -0.12; d[5] = -0.12
    const v = votes(30, 0); v[0] = 4; v[4] = 0.5; v[5] = 3
    const r = knockoutRoles(sweep(d), v)
    expect(r).toContainEqual({ layer: 4, role: 'preparatory' })
    expect(r.map((x) => x.layer)).not.toContain(5)
  })
  it('no votes at all: nothing is compensated, small votes can be preparatory', () => {
    const d = flat(); d[1] = 0.01; d[2] = -0.4
    expect(knockoutRoles(sweep(d), votes(30, 0))).toEqual([{ layer: 2, role: 'preparatory' }])
  })
  it('at most 5, decisive first then by |dp|', () => {
    const d = flat()
    d[0] = -0.2; d[1] = -0.3; d[2] = -0.4; d[3] = -0.25; d[4] = -0.35; d[5] = -0.5; d[6] = 0.2
    const v = votes(30, 0); v[29] = 5
    const r = knockoutRoles(sweep(d, [0]), v)
    expect(r).toHaveLength(MAX_ROLE_CHIPS)
    expect(r.map((x) => x.layer)).toEqual([0, 5, 2, 4, 1])
    expect(r[0].role).toBe('decisive')
  })
})

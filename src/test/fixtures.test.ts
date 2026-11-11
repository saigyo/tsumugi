import { expect, test } from 'vitest'
import { sweepProblems } from '../trace/validate'
import { knockoutRoles } from '../viz/knockoutRoles'
import { buildFixtureTrace, fixtureSweep } from './fixtures'

test('the fixture sweep is valid and shows a decisive, a compensated and a dampening layer and a flip', () => {
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true, sweepable: true })
  expect(events[0]).toMatchObject({ type: 'run-start', seed: 7, revision: 'v7' })
  for (const cycle of [0, 1]) {
    const sweep = fixtureSweep(events, cycle)
    expect(sweepProblems(sweep, 3)).toEqual([])
    expect(sweep.layers.some((l) => l.flips)).toBe(true)
    const choice = events.find((e) => e.type === 'logits' && e.cycle === cycle)
    const sample = events.find((e) => e.type === 'sample' && e.cycle === cycle)
    if (choice?.type !== 'logits' || sample?.type !== 'sample') throw new Error('incomplete cycle')
    const votes = choice.topK.find((c) => c.id === sample.chosen.id)!.votes!.mlp
    expect(knockoutRoles(sweep, votes).map((r) => r.role).sort()).toEqual(['compensated', 'dampening', 'decisive'])
  }
})

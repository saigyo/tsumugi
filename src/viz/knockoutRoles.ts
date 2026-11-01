import type { KnockoutSweep } from '../trace/types'

export type KnockoutRole = 'decisive' | 'preparatory' | 'compensated' | 'dampening'

export interface RoleThresholds {
  drop: number // probability drop that counts as a real loss
  rise: number // probability rise that counts as dampening
  smallVote: number // fraction of the largest |vote| at or below which a vote is small
  largeVote: number // fraction of the largest |vote| at or above which a vote is large
}

// Provisional values; calibrated against real sweeps later
export const ROLE_THRESHOLDS: RoleThresholds = { drop: 0.15, rise: 0.05, smallVote: 0.25, largeVote: 1.0 }

export const MAX_ROLE_CHIPS = 5

export function knockoutRoles(
  sweep: KnockoutSweep,
  votes: number[],
  t: RoleThresholds = ROLE_THRESHOLDS,
): Array<{ layer: number; role: KnockoutRole }> {
  const max = Math.max(0, ...votes.map(Math.abs))
  const out: Array<{ layer: number; role: KnockoutRole; mag: number }> = []
  for (const l of sweep.layers) {
    const dp = l.prob - sweep.baseline
    const vote = Math.abs(votes[l.layer] ?? 0)
    const large = max > 0 && vote >= t.largeVote * max
    let role: KnockoutRole | null = null
    if (l.flips) role = 'decisive'
    else if (dp >= t.rise) role = 'dampening'
    else if (dp <= -t.drop && vote <= t.smallVote * max) role = 'preparatory'
    else if (large && Math.abs(dp) < t.rise) role = 'compensated'
    if (role) out.push({ layer: l.layer, role, mag: Math.abs(dp) })
  }
  out.sort((a, b) => Number(b.role === 'decisive') - Number(a.role === 'decisive') || b.mag - a.mag)
  return out.slice(0, MAX_ROLE_CHIPS).map(({ layer, role }) => ({ layer, role }))
}

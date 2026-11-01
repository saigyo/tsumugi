import type { KnockoutSweep } from '../trace/types'

export type KnockoutRole = 'decisive' | 'preparatory' | 'compensated' | 'dampening'

export interface RoleThresholds {
  drop: number // probability drop that counts as a real loss
  rise: number // probability rise that counts as dampening
  smallVote: number // fraction of the largest |vote| at or below which a vote is small
  largeVote: number // fraction of the largest |vote| at or above which a vote is large
}

// Calibrated on ten real SmolLM2-135M v7 sweeps, see docs/research/2026-11-01-knockout-roles-spike.md
export const ROLE_THRESHOLDS: RoleThresholds = { drop: 0.10, rise: 0.10, smallVote: 0.25, largeVote: 0.5 }

export const MAX_ROLE_CHIPS = 5
// decisive layers are common on a small model; keep room for the other roles
export const MAX_DECISIVE_CHIPS = 3

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
  const byMag = (a: { mag: number }, b: { mag: number }) => b.mag - a.mag
  // removing layer 0 always breaks the model, so it says least: it sorts last among the decisive layers
  const decisive = out.filter((x) => x.role === 'decisive')
    .sort((a, b) => (a.layer === 0 ? 1 : 0) - (b.layer === 0 ? 1 : 0) || byMag(a, b))
  const others = out.filter((x) => x.role !== 'decisive').sort(byMag)
  // up to 3 decisive plus the other roles; unused slots go back to further decisive layers
  const nDecisive = Math.min(decisive.length, Math.max(MAX_DECISIVE_CHIPS, MAX_ROLE_CHIPS - others.length))
  const nOthers = Math.min(others.length, MAX_ROLE_CHIPS - nDecisive)
  return [...decisive.slice(0, nDecisive), ...others.slice(0, nOthers)].map(({ layer, role }) => ({ layer, role }))
}

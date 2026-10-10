import type { LayerVotes } from '../trace/types'

// Numbers behind the MLP cards. Kept apart from the components so those
// files export only components.

// A candidate's score building up through the model: the embedding's share,
// then per layer its attention vote followed by its MLP vote.
export function runningTotal(v: LayerVotes): number[] {
  const out = [v.embed]
  let s = v.embed
  v.attn.forEach((a, l) => { s += a; out.push(s); s += v.mlp[l]; out.push(s) })
  return out
}

const sum = (a: number[]) => a.reduce((s, x) => s + x, 0)

export const blockTotals = (v: LayerVotes) => ({ embed: v.embed, attn: sum(v.attn), mlp: sum(v.mlp) })

export function topMlpLayers(v: LayerVotes, n = 3): Array<{ layer: number; value: number }> {
  return v.mlp.map((value, layer) => ({ layer, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, n)
}

// One symmetric colour scale for the whole grid, so cells compare across candidates.
export const votePeak = (cands: Array<{ votes: LayerVotes }>) =>
  cands.reduce((m, c) => Math.max(m, ...c.votes.mlp.map(Math.abs)), 0) || 1

// Edit size 0 → pale, ≥ 1 → deep indigo (the attention heatmap's ramp).
export function ratioFill(v: number): string {
  const t = Math.min(1, Math.max(0, v))
  return `hsl(211 ${Math.round(30 + 25 * t)}% ${Math.round(94 - 70 * t)}%)`
}

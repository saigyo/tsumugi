import type { AttentionHead, TokenInfo } from '../../trace/types'

// Deterministic, hand-shaped attention patterns for simulated mode. Each row i
// covers positions 0..i (causal, ragged) and sums to exactly 1 by construction.

function row(i: number, weights: Array<[number, number]>): number[] {
  const w = Array.from({ length: i + 1 }, () => 0)
  for (const [pos, mass] of weights) w[pos] += mass
  return w
}

export function prevTokenRow(i: number): number[] {
  if (i === 0) return [1]
  return row(i, [[i - 1, 0.8], [i, 0.2]])
}

export function sinkRow(i: number): number[] {
  if (i === 0) return [1]
  return row(i, [[0, 0.7], [i - 1, 0.2], [i, 0.1]])
}

export function inductionRow(i: number, tokens: TokenInfo[]): number[] {
  if (i === 0) return [1]
  for (let j = i - 1; j >= 0; j--) {
    if (tokens[j].text.trim() === tokens[i].text.trim() && j + 1 <= i) {
      return row(i, [[j + 1, 0.6], [0, 0.2], [i - 1, 0.2]])
    }
  }
  return row(i, [[i - 1, 0.5], [0, 0.3], [i, 0.2]])
}

const matrix = (n: number, rowFor: (i: number) => number[]): number[][] =>
  Array.from({ length: n }, (_, i) => rowFor(i))

export function proceduralHeads(tokens: TokenInfo[]): AttentionHead[] {
  const n = tokens.length
  return [
    { layer: 0, head: 3, label: 'attention-sink', matrix: matrix(n, sinkRow) },
    { layer: 2, head: 1, label: 'previous-token', matrix: matrix(n, prevTokenRow) },
    { layer: 5, head: 7, label: 'induction', matrix: matrix(n, (i) => inductionRow(i, tokens)) },
  ]
}

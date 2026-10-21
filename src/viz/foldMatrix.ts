// A ragged causal attention matrix with a chat run's system turn folded: the
// span's rows hidden, its columns summed into one column right where it
// starts. Row 0 (the first <|im_start|>, the attention sink) stays.
export interface FoldedMatrix {
  rows: number[]                              // original row positions, in order
  cols: Array<{ from: number; to: number }>   // half-open position ranges, one per shown column
  values: number[][]                          // values[i][j] for every column with from <= rows[i]
}

export function foldMatrix(matrix: number[][], system: [number, number] | undefined): FoldedMatrix {
  const n = matrix.length
  const fold = system && system[0] < system[1] && system[1] <= n ? system : undefined
  const rows = Array.from({ length: n }, (_, i) => i).filter((i) => !fold || i < fold[0] || i >= fold[1])
  const cols: Array<{ from: number; to: number }> = []
  for (let i = 0; i < n; i++) {
    if (fold && i === fold[0]) { cols.push({ from: fold[0], to: fold[1] }); i = fold[1] - 1; continue }
    cols.push({ from: i, to: i + 1 })
  }
  const values = rows.map((r) => cols.filter((c) => c.from <= r).map((c) => {
    let sum = 0
    for (let k = c.from; k < Math.min(c.to, r + 1); k++) sum += matrix[r][k] ?? 0
    return sum
  }))
  return { rows, cols, values }
}

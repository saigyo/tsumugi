// Turns the model's `inputs_embeds` output ([1, fed, dims], the embedding
// lookup for the tokens fed this cycle) into trace rows. Pure, so the shape
// policy is unit-testable outside the worker.
export type EmbedRowsResult =
  | { status: 'absent' }
  | { status: 'bad-shape'; dims: number[] }
  | { status: 'ok'; rows: number[][] }

export function extractEmbedRows(
  tensor: { dims: number[]; data: ArrayLike<number> } | undefined,
  fed: number, dims: number, decimals = 3,
): EmbedRowsResult {
  if (!tensor) return { status: 'absent' }
  const d = tensor.dims
  if (d.length !== 3 || d[0] !== 1 || d[1] !== fed || d[2] !== dims || tensor.data.length !== fed * dims)
    return { status: 'bad-shape', dims: [...d] }
  const f = 10 ** decimals
  const rows: number[][] = []
  for (let r = 0; r < fed; r++) {
    const row = new Array<number>(dims)
    for (let c = 0; c < dims; c++) row[c] = Math.round(tensor.data[r * dims + c] * f) / f
    rows.push(row)
  }
  return { status: 'ok', rows }
}

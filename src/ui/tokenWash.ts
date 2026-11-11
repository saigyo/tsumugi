const WASHES = 6

// The wash (0 to 5) of each prompt token, given out by word in order of first
// appearance: a token id seen before reuses its wash; a new one takes the next
// wash, skipping the wash of the position just before it. Prefix-stable: the
// washes of the first n ids do not depend on what follows, so a card that
// shows a longer list (prompt plus generated tokens) agrees with one that
// shows only the prompt.
export function washIndexes(ids: number[]): number[] {
  const seen = new Map<number, number>()
  const out: number[] = []
  let next = 0
  for (const id of ids) {
    let wash = seen.get(id)
    if (wash === undefined) {
      wash = next % WASHES
      if (out.length > 0 && wash === out[out.length - 1]) wash = (wash + 1) % WASHES
      seen.set(id, wash)
      next = wash + 1
    }
    out.push(wash)
  }
  return out
}

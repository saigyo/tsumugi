// Newest first: the `limit` newest runs plus any of `keep` that are older, each in its place.
export function shelfRow<T extends { id: string }>(
  oldestFirst: T[], keep: Array<string | null | undefined>, limit = 5,
): { row: T[]; hidden: number } {
  const newestFirst = [...oldestFirst].reverse()
  const kept = new Set(keep.filter((k): k is string => !!k))
  const row = newestFirst.filter((r, i) => i < limit || kept.has(r.id))
  return { row, hidden: newestFirst.length - row.length }
}

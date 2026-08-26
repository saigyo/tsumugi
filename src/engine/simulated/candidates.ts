const COMMON = [
  'the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'he', 'for', 'it',
  'with', 'as', 'his', 'on', 'be', 'at', 'by', 'had', 'not', 'are', 'but',
  'from', 'or', 'have', 'an', 'they', 'which', 'one', 'you', 'were', 'her',
  'all', 'she', 'there', 'would', 'their', 'we', 'him', 'been', 'has', 'when',
  'who', 'will', 'more', 'no', 'if', 'out', 'so',
]

export function candidateWords(prevText: string, rand: () => number): string[] {
  const sentenceStart = /[.!?]\s*$/.test(prevText) || prevText.trim() === ''
  const sinceEnd = prevText.length - Math.max(
    prevText.lastIndexOf('.'), prevText.lastIndexOf('!'), prevText.lastIndexOf('?'))

  const pool = [...COMMON]
  const picks: string[] = []
  while (picks.length < 9 && pool.length > 0) {
    const w = pool.splice(Math.floor(rand() * pool.length), 1)[0]
    picks.push(' ' + (sentenceStart ? w[0].toUpperCase() + w.slice(1) : w))
  }
  picks.push(sinceEnd > 40 ? '.' : ',')
  return picks
}

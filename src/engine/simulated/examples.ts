import type { AttentionHead, TokenInfo } from '../../trace/types'
import { prevTokenRow, proceduralHeads } from './attention'

export interface CuratedExample {
  id: string
  label: string
  prompt: string
  hint: string
}

export const CURATED_EXAMPLES: CuratedExample[] = [
  {
    id: 'coreference',
    label: 'Coreference: “it” → “cat”',
    prompt: 'The cat sat on the mat because it was tired',
    hint: 'Open the coreference head and find the row for “it”: most of its attention points back to “cat” — the model resolving what the pronoun refers to.',
  },
  {
    id: 'induction',
    label: 'Induction: repeated pattern',
    prompt: 'one two three one two three one',
    hint: 'Open the induction head: on each repeated token, attention jumps to what followed its previous occurrence — the circuit behind in-context pattern completion.',
  },
  {
    id: 'sink',
    label: 'Attention sink',
    prompt: 'The quick brown fox jumps over the lazy dog',
    hint: 'Open the attention-sink head: almost every row lights up the first column — surplus attention parking on the first token as a learned “do nothing” default.',
  },
]

function coreferenceHead(tokens: TokenInfo[]): AttentionHead | null {
  const norm = (t: TokenInfo) => t.text.trim().toLowerCase()
  const itIdx = tokens.findIndex((t) => norm(t) === 'it')
  const catIdx = tokens.findIndex((t) => norm(t) === 'cat')
  if (itIdx < 0 || catIdx < 0 || catIdx >= itIdx) return null
  return {
    layer: 8, head: 2, label: 'coreference',
    matrix: tokens.map((_, i) => {
      if (i !== itIdx) return prevTokenRow(i)
      const w = Array.from({ length: i + 1 }, () => 0)
      w[catIdx] += 0.65
      w[0] += 0.15
      w[i - 1] += 0.2
      return w
    }),
  }
}

export function attentionHeadsFor(prompt: string, tokens: TokenInfo[]): AttentionHead[] {
  const heads = proceduralHeads(tokens)
  const example = CURATED_EXAMPLES.find((e) => e.prompt === prompt.trim())
  if (example?.id === 'coreference') {
    const coref = coreferenceHead(tokens)
    if (coref) heads.push(coref)
  }
  return heads
}

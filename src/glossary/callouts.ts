import type { GlossaryId } from './types'

// The ⓘ chips on the cards: label as shown, and the entry section it opens.
export interface CalloutTarget { label: string; id: GlossaryId; section: string }

export const EMBEDDING_CALLOUTS: CalloutTarget[] = [
  { label: 'learned, not designed', id: 'embedding', section: 'learned' },
  { label: 'no position here', id: 'embedding', section: 'position' },
  { label: 'tied with Logits', id: 'embedding', section: 'tied' },
]

export const logitsCallouts = (shown: number, vocab: string): CalloutTarget[] => [
  { label: 'tied with Embeddings', id: 'logit', section: 'tied' },
  { label: 'similar rows, similar scores', id: 'logit', section: 'similar-rows' },
  { label: `${vocab} scores, ${shown} shown`, id: 'logit', section: 'top-k' },
]

export const SOFTMAX_CALLOUTS: CalloutTarget[] = [
  { label: 'why exponentiate', id: 'softmax', section: 'exp' },
  { label: 'temperature', id: 'softmax', section: 'temperature' },
  { label: 'top-k only', id: 'softmax', section: 'top-k' },
]

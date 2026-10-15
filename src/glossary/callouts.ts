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

export const EDIT_CALLOUTS: CalloutTarget[] = [
  { label: 'why layer 0 is so dark', id: 'edit', section: 'layer-0' },
  { label: 'quiet middle, loud end', id: 'edit', section: 'quiet-middle' },
  { label: 'attention vs MLP', id: 'edit', section: 'attn-vs-mlp' },
]

export const NEURON_CALLOUTS: CalloutTarget[] = [
  { label: 'why the list rarely names the candidate', id: 'mlp-neuron', section: 'rarely-names' },
  { label: 'negative activations', id: 'mlp-neuron', section: 'negative' },
  { label: 'early layers', id: 'mlp-neuron', section: 'early-layers' },
]

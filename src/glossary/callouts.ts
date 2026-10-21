import type { GlossaryId } from './types'

// The ⓘ chips on the cards: label as shown, the entry section it opens, and
// the hover sentence that says what that section answers.
export interface CalloutTarget { label: string; id: GlossaryId; section: string; tip: string }

export const EMBEDDING_CALLOUTS: CalloutTarget[] = [
  { label: 'learned, not designed', id: 'embedding', section: 'learned',
    tip: 'Nobody designed these rows: training adjusts them, so tokens used alike end up pointing alike.' },
  { label: 'no position here', id: 'embedding', section: 'position',
    tip: '“cat” gets the same row wherever it appears; position enters later, inside attention.' },
  { label: 'tied with Logits', id: 'embedding', section: 'tied',
    tip: 'The same matrix scores every token at the end of the pipeline.' },
]

export const logitsCallouts = (shown: number, vocab: string): CalloutTarget[] => [
  { label: 'tied with Embeddings', id: 'logit', section: 'tied',
    tip: 'The Embeddings matrix read the other way: every row is scored against the final vector.' },
  { label: 'similar rows, similar scores', id: 'logit', section: 'similar-rows',
    tip: 'Tokens whose rows point the same way get nearly the same score, so near-synonyms sit together.' },
  { label: `${vocab} scores, ${shown} shown`, id: 'logit', section: 'top-k',
    tip: 'Every vocabulary entry gets a score; the trace keeps only the top-k the sampler can pick from.' },
]

export const SOFTMAX_CALLOUTS: CalloutTarget[] = [
  { label: 'why exponentiate', id: 'softmax', section: 'exp',
    tip: 'exp makes every score positive and turns a lead of a few points into a large ratio.' },
  { label: 'temperature', id: 'softmax', section: 'temperature',
    tip: 'Dividing the scores by T sharpens the leads (T below 1) or flattens them (T above 1).' },
  { label: 'top-k only', id: 'softmax', section: 'top-k',
    tip: 'The percentages are shares among the kept candidates; every other token gets nothing.' },
]

export const EDIT_CALLOUTS: CalloutTarget[] = [
  { label: 'why layer 0 is so dark', id: 'edit', section: 'layer-0',
    tip: 'The stream starts small, so the first edits are large next to it. That does not make them the most important.' },
  { label: 'quiet middle, loud end', id: 'edit', section: 'quiet-middle',
    tip: 'Middle edits look quiet next to a grown stream; the last layers write large edits for the readout.' },
  { label: 'attention vs MLP', id: 'edit', section: 'attn-vs-mlp',
    tip: 'Attention moves information between positions; the MLP works on each position alone.' },
]

export const NEURON_CALLOUTS: CalloutTarget[] = [
  { label: 'why the list rarely names the candidate', id: 'mlp-neuron', section: 'rarely-names',
    tip: 'A vote counts only the part of a neuron’s push that points along the candidate’s row.' },
  { label: 'negative activations', id: 'mlp-neuron', section: 'negative',
    tip: 'A negative activation adds the neuron’s column flipped, pushing toward a different set of tokens.' },
  { label: 'early layers', id: 'mlp-neuron', section: 'early-layers',
    tip: 'Early votes are small, and their token lists look like noise.' },
]

export const MLP_STEP_CALLOUTS: CalloutTarget[] = [
  { label: 'why SiLU and not a switch', id: 'mlp', section: 'silu-not-switch',
    tip: 'SiLU is smooth: negative gates give small values instead of a hard zero, so training can still adjust them.' },
  { label: 'why two projections', id: 'mlp', section: 'two-projections',
    tip: 'The gate decides how much passes, the up value what passes and with which sign.' },
  { label: 'early layers look different', id: 'mlp', section: 'early-layers',
    tip: 'Early layers spread the activation over about a hundred neurons; the last layers over about ten.' },
]

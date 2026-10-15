import { attention } from './entries/attention'
import { edit } from './entries/edit'
import { embedding } from './entries/embedding'
import { howToRead } from './entries/how-to-read'
import { layer } from './entries/layer'
import { logit } from './entries/logit'
import { mlp } from './entries/mlp'
import { mlpNeuron } from './entries/mlp-neuron'
import { modes } from './entries/modes'
import { residualStream } from './entries/residual-stream'
import { sampling } from './entries/sampling'
import { softmax } from './entries/softmax'
import { token } from './entries/token'
import { votes } from './entries/votes'
import type { GlossaryEntry, GlossaryId, GlossaryStage } from './types'

export const ENTRIES: GlossaryEntry[] = [
  howToRead, token, embedding, residualStream, edit, layer, attention, mlp, votes, mlpNeuron, logit, softmax, sampling, modes,
]

const BY_ID = new Map<string, GlossaryEntry>(ENTRIES.map((e) => [e.id, e]))

// Looked up at render time (never at module load): entries import Term, and
// Term imports this module.
export const getEntry = (id: string): GlossaryEntry | undefined => BY_ID.get(id)

export const STAGE_ORDER: GlossaryStage[] = ['tokenize', 'embed', 'layers', 'logits', 'softmax', 'sample', 'general']

export const STAGE_LABEL: Record<GlossaryStage, string> = {
  tokenize: 'Tokenizer', embed: 'Embeddings', layers: 'Layers', logits: 'Logits',
  softmax: 'Softmax', sample: 'Sampler', general: 'General',
}

const FIRST: GlossaryId = 'how-to-read'

// The index below "How to read Tsumugi": entries grouped by stage in pipeline
// order, registry order within a group, empty groups left out.
export function indexGroups(): Array<{ stage: GlossaryStage; label: string; entries: GlossaryEntry[] }> {
  return STAGE_ORDER
    .map((stage) => ({ stage, label: STAGE_LABEL[stage], entries: ENTRIES.filter((e) => e.stage === stage && e.id !== FIRST) }))
    .filter((g) => g.entries.length > 0)
}

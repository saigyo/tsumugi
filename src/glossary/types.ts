import type { ReactNode } from 'react'
import type { GenParams, Mode, RunEndReason, TokenInfo } from '../trace/types'

// The ids are a plain list, not derived from the entry objects: entries refer
// to each other (related, <Term>), so deriving the type from them would be
// circular. The registry test checks one entry per id.
export const GLOSSARY_IDS = [
  'how-to-read', 'token', 'chat-template', 'embedding', 'residual-stream', 'edit', 'layer',
  'attention', 'mlp', 'votes', 'knockout', 'mlp-neuron', 'logit', 'softmax', 'sampling', 'generation-settings', 'modes',
] as const
export type GlossaryId = typeof GLOSSARY_IDS[number]

export type GlossaryStage = 'general' | 'tokenize' | 'embed' | 'layers' | 'logits' | 'softmax' | 'sample'

// What "In this run" blocks get to work with: plain values derived from the
// active trace at the cycle the player shows.
export interface RunSummary {
  mode: Mode
  modelId: string
  params: GenParams
  tokens: TokenInfo[]                    // the prompt's tokens
  cycle: number                          // viewed cycle, 0-based
  cycles: number                         // cycles in the trace so far
  generated: number                      // tokens appended in the whole trace
  ended?: RunEndReason                   // absent while the run is still going
  stream?: { seqLen: number; dims: number; layers: number }
  vocabSize?: number
  softmax?: { temperature: number; top: Array<{ id: number; text: string; logit?: number; prob: number }> }
  edits?: { token: TokenInfo; layer: number; ratio: number }      // largest MLP edit of the last visible token
  votes?: { candidate: TokenInfo; layer: number; value: number }  // MLP layer adding most to the top candidate
  // the saved knockout sweep for the viewed cycle; drop and rise are probability changes in 0..1
  knockout?: { chosen: TokenInfo; baseline: number; layers: number; flipped: number
    drop?: { layer: number; delta: number }; rise?: { layer: number; delta: number } }
}

export interface GlossaryEntry {
  id: GlossaryId
  title: string
  summary: string
  stage: GlossaryStage
  related: GlossaryId[]
  body: () => ReactNode
  inRun?: (run: RunSummary) => ReactNode | null
}

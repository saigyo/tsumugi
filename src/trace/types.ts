export interface TokenInfo { id: number; text: string }
export interface GenParams { temperature: number; topK: number; maxNewTokens: number }
export type Mode = 'sim' | 'real'

export type AttentionLabel = 'previous-token' | 'attention-sink' | 'induction' | 'coreference' | 'distinctive' | 'pinned'

export interface AttentionHead {
  layer: number
  head: number
  label: AttentionLabel
  score?: number
  // ragged causal matrix: row i holds weights for positions 0..i and sums to 1
  matrix: number[][]
}

export interface AttentionGridCell {
  layer: number
  head: number
  // ≤12×12 mean-pooled thumbnail of the accumulated causal matrix, values 0..1.
  // Mean pooling preserves relative mass; pooled rows do NOT sum to 1.
  thumb: number[][]
  prevTokenScore: number
  sinkScore: number
  inductionScore: number | null
  // optional: traces recorded before the coreference detector lack it
  corefScore?: number | null
  distinctiveScore: number
}

export type RunEndReason = 'eos' | 'max-tokens' | 'aborted' | 'error'

// Where a run's embedding vectors come from: 'model' = exact rows emitted into
// the trace (real mode with the inputs_embeds export); 'asset' = the view looks
// vectors up by id in the Hub geometry asset (sim mode, or an old cached model).
export type EmbedSource = 'model' | 'asset'

export type TraceEvent =
  | { type: 'run-start'; prompt: string; mode: Mode; modelId: string; params: GenParams; vocabSize?: number }
  | { type: 'tokenize'; tokens: TokenInfo[]; truncated?: boolean }
  | { type: 'embed'; cycle: number; seqLen: number; dims: number; source: EmbedSource; rows?: number[][] }
  | { type: 'layer'; cycle: number; index: number; total: number; activationNorm?: number }
  | { type: 'attention'; cycle: number; heads: AttentionHead[] }
  | { type: 'logits'; cycle: number; topK: Array<TokenInfo & { logit: number }> }
  | { type: 'softmax'; cycle: number; temperature: number; topK: Array<TokenInfo & { prob: number }> }
  | { type: 'sample'; cycle: number; chosen: TokenInfo; method: 'greedy' | 'top-k' }
  | { type: 'append'; cycle: number; token: TokenInfo }
  | { type: 'attention-grid'; layers: number; heads: number; cells: AttentionGridCell[] }
  | { type: 'run-end'; reason: RunEndReason; message?: string }

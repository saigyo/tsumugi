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
export type RunEndReason = 'eos' | 'max-tokens' | 'aborted' | 'error'
export type TraceEvent =
  | { type: 'run-start'; prompt: string; mode: Mode; modelId: string; params: GenParams; vocabSize?: number }
  | { type: 'tokenize'; tokens: TokenInfo[]; truncated?: boolean }
  | { type: 'embed'; cycle: number; seqLen: number; dims: number; preview: number[][] }
  | { type: 'layer'; cycle: number; index: number; total: number; activationNorm?: number }
  | { type: 'attention'; cycle: number; heads: AttentionHead[] }
  | { type: 'logits'; cycle: number; topK: Array<TokenInfo & { logit: number }> }
  | { type: 'softmax'; cycle: number; temperature: number; topK: Array<TokenInfo & { prob: number }> }
  | { type: 'sample'; cycle: number; chosen: TokenInfo; method: 'greedy' | 'top-k' }
  | { type: 'append'; cycle: number; token: TokenInfo }
  | { type: 'run-end'; reason: RunEndReason; message?: string }

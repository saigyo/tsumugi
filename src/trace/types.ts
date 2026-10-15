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

// The readout strips: the final vector and E rows pooled into this many cells
// (576 / 64 = 9 dimensions per cell). Shared by the worker's pooling and the
// Logits card's pooling of lastHidden, so the strips line up cell for cell.
export const READOUT_CELLS = 64

// Real mode with the v4 export: one candidate's score split into the
// embedding's share and one share per attention and MLP block, each part
// scaled by the final norm, so embed + Σ attn + Σ mlp = the logit.
export interface LayerVotes { embed: number; attn: number[]; mlp: number[] }

export interface LogitCandidate extends TokenInfo {
  logit: number
  // real mode with the v3 export: the candidate's row of E pooled by mean,
  // and the element-wise product x_last ⊙ E[id] pooled by sum (Σ dot ≈ logit)
  row?: number[]
  dot?: number[]
  votes?: LayerVotes
}

// v5 export, real mode: the last token's neuron activations for every layer
// and what the browser needs to split an MLP vote into neuron votes. Base64
// keeps it to about 150 KB per cycle (see trace/neuronPack.ts).
export interface NeuronData {
  neurons: number                                   // per layer
  activations: string                               // float16 [layers × neurons], layer-major
  invRms: number                                    // the final norm's 1/rms at the last position
  candidates: Array<{ id: number; row: string }>    // each candidate's row of E, float32 [dims]
}

export type TraceEvent =
  | { type: 'run-start'; prompt: string; mode: Mode; modelId: string; params: GenParams; vocabSize?: number }
  | { type: 'tokenize'; tokens: TokenInfo[]; truncated?: boolean }
  | { type: 'embed'; cycle: number; seqLen: number; dims: number; source: EmbedSource; rows?: number[][] }
  | { type: 'layer'; cycle: number; index: number; total: number; activationNorm?: number }
  | { type: 'attention'; cycle: number; heads: AttentionHead[] }
  // v4 export: ‖edit‖/‖stream before‖ per fed position (rows start at `start`) and layer;
  // v5 adds the last token's neuron data
  | { type: 'edits'; cycle: number; start: number; attn: number[][]; mlp: number[][]; neurons?: NeuronData }
  | { type: 'logits'; cycle: number; topK: LogitCandidate[]; lastHidden?: number[] }
  | { type: 'softmax'; cycle: number; temperature: number; topK: Array<TokenInfo & { prob: number }> }
  | { type: 'sample'; cycle: number; chosen: TokenInfo; method: 'greedy' | 'top-k' }
  | { type: 'append'; cycle: number; token: TokenInfo }
  | { type: 'attention-grid'; layers: number; heads: number; cells: AttentionGridCell[] }
  | { type: 'run-end'; reason: RunEndReason; message?: string }

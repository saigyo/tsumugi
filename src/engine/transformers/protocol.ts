import type { AttentionLabel, GenParams, TraceEvent } from '../../trace/types'
import type { ProgressInfo } from '../types'

export type WorkerRequest =
  | { type: 'prepare'; modelId: string }
  | { type: 'run'; runId: number; prompt: string; params: GenParams }
  | { type: 'abort' }
  // pin-exactness side channel — the one data path beside the trace;
  // must not grow other uses without a new design
  | { type: 'head-request'; layer: number; head: number }
export type WorkerResponse =
  | { type: 'progress'; info: ProgressInfo }
  | { type: 'ready'; device: 'webgpu' | 'wasm'; attentions: boolean }
  | { type: 'trace'; runId: number; event: TraceEvent }
  | { type: 'done'; runId: number }
  | { type: 'fatal'; message: string }
  | { type: 'head-response'; layer: number; head: number
      matrix: number[][]                // exact accumulated ragged matrix; [] if unavailable
      label: AttentionLabel | null      // best template ≥ 0.3, else null
      score: number | null }

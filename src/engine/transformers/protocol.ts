import type { AttentionLabel, GenParams, KnockoutLayer, PromptFormat, TraceEvent } from '../../trace/types'
import type { ProgressInfo, RunOptions } from '../types'

export type WorkerRequest =
  | { type: 'prepare'; modelId: string }
  | { type: 'run'; runId: number; prompt: string; params: GenParams; format: PromptFormat
      knockout?: RunOptions['knockout'] }
  | { type: 'abort' }
  // side channel beside the trace, with exactly two uses: head pins (pin-exactness)
  // and knockout sweeps (design doc section 3); must not grow without a new design
  | { type: 'head-request'; layer: number; head: number }
  | { type: 'sweep-request'; sweepId: number; ids: number[]; params: GenParams
      chosenId: number; seed: number; cycle: number }
export type WorkerResponse =
  | { type: 'progress'; info: ProgressInfo }
  | { type: 'ready'; device: 'webgpu' | 'wasm'; attentions: boolean
      knockout: boolean; revision?: string }
  | { type: 'trace'; runId: number; event: TraceEvent }
  | { type: 'done'; runId: number }
  | { type: 'sweep-baseline'; sweepId: number; prob: number }
  | { type: 'sweep-layer'; sweepId: number; entry: KnockoutLayer }
  | { type: 'sweep-done'; sweepId: number; aborted: boolean }
  | { type: 'fatal'; message: string }
  | { type: 'head-response'; layer: number; head: number
      matrix: number[][]                // exact accumulated ragged matrix; [] if unavailable
      label: AttentionLabel | null      // best template ≥ 0.3, else null
      score: number | null }

import type { GenParams, TraceEvent } from '../../trace/types'
import type { ProgressInfo } from '../types'

export type WorkerRequest =
  | { type: 'prepare'; modelId: string }
  | { type: 'run'; runId: number; prompt: string; params: GenParams }
  | { type: 'abort' }
export type WorkerResponse =
  | { type: 'progress'; info: ProgressInfo }
  | { type: 'ready'; device: 'webgpu' | 'wasm'; attentions: boolean }
  | { type: 'trace'; runId: number; event: TraceEvent }
  | { type: 'done'; runId: number }
  | { type: 'fatal'; message: string }

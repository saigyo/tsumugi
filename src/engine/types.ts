import type { GenParams, KnockoutOrigin, PromptFormat, TraceEvent } from '../trace/types'

export interface ProgressInfo { file: string; loaded: number; total: number }
export interface RunHandle { abort(): void; done: Promise<void> }
export interface RunOptions { knockout?: { origin: KnockoutOrigin; forced: number[]; seed: number } }
export interface PipelineEngine {
  prepare(onProgress?: (p: ProgressInfo) => void): Promise<void>
  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void, format?: PromptFormat, options?: RunOptions): RunHandle
}

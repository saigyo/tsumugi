import type { AttentionLabel, GenParams, TraceEvent } from '../../trace/types'
import { MODEL_ID } from '../tokenizer'
import type { PipelineEngine, ProgressInfo, RunHandle } from '../types'
import type { WorkerRequest, WorkerResponse } from './protocol'

export interface HeadData {
  layer: number
  head: number
  matrix: number[][]
  label: AttentionLabel | null
  score: number | null
}

export class TransformersEngine implements PipelineEngine {
  private worker: Worker
  private nextRunId = 1
  device: 'webgpu' | 'wasm' | null = null
  attentions: boolean | null = null
  private listeners = new Set<(msg: WorkerResponse) => void>()

  constructor(workerFactory: () => Worker = () =>
    new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })) {
    this.worker = workerFactory()
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      for (const l of [...this.listeners]) l(e.data)
    }
  }

  private post(msg: WorkerRequest) { this.worker.postMessage(msg) }

  prepare(onProgress?: (p: ProgressInfo) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      const listener = (msg: WorkerResponse) => {
        if (msg.type === 'progress') onProgress?.(msg.info)
        if (msg.type === 'ready') { this.device = msg.device; this.attentions = msg.attentions; this.listeners.delete(listener); resolve() }
        if (msg.type === 'fatal') { this.listeners.delete(listener); reject(new Error(msg.message)) }
      }
      this.listeners.add(listener)
      this.post({ type: 'prepare', modelId: MODEL_ID })
    })
  }

  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void): RunHandle {
    const runId = this.nextRunId++
    const done = new Promise<void>((resolve) => {
      const listener = (msg: WorkerResponse) => {
        if (msg.type === 'trace' && msg.runId === runId) emit(msg.event)
        if (msg.type === 'done' && msg.runId === runId) { this.listeners.delete(listener); resolve() }
        if (msg.type === 'fatal') {
          emit({ type: 'run-end', reason: 'error', message: msg.message })
          this.listeners.delete(listener); resolve()
        }
      }
      this.listeners.add(listener)
      this.post({ type: 'run', runId, prompt, params })
    })
    return { abort: () => this.post({ type: 'abort' }), done }
  }

  fetchHead(layer: number, head: number): Promise<HeadData> {
    return new Promise((resolve) => {
      const listener = (msg: WorkerResponse) => {
        if (msg.type === 'head-response' && msg.layer === layer && msg.head === head) {
          this.listeners.delete(listener)
          resolve({ layer: msg.layer, head: msg.head, matrix: msg.matrix, label: msg.label, score: msg.score })
        }
      }
      this.listeners.add(listener)
      this.post({ type: 'head-request', layer, head })
    })
  }
}

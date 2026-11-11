import type { AttentionLabel, GenParams, KnockoutLayer, PromptFormat, TraceEvent } from '../../trace/types'
import { TEXT_FORMAT } from '../chatTemplate'
import { MODEL_ID } from '../tokenizer'
import type { PipelineEngine, ProgressInfo, RunHandle, RunOptions } from '../types'
import type { WorkerRequest, WorkerResponse } from './protocol'

// ok: the sweep ran to the end; not ok without an error: aborted; with an error: failed
export interface SweepResult { ok: boolean; error?: string }

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
  private nextSweepId = 1
  device: 'webgpu' | 'wasm' | null = null
  attentions: boolean | null = null
  knockout: boolean | null = null
  revision: string | null = null
  private listeners = new Set<(msg: WorkerResponse) => void>()

  private workerFactory: () => Worker

  constructor(workerFactory: () => Worker = () =>
    new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })) {
    this.workerFactory = workerFactory
    this.worker = this.spawn()
  }

  private spawn(): Worker {
    const worker = this.workerFactory()
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      if (worker !== this.worker) return // a worker that was replaced has nothing more to say
      for (const l of [...this.listeners]) l(e.data)
    }
    return worker
  }

  private post(msg: WorkerRequest) { this.worker.postMessage(msg) }

  // The worker tries one entry of its loading plan (the custom export or the stock one, on the graphics card
  // or the CPU). When that fails, the next entry is tried in a fresh worker: after one failed session load,
  // transformers.js (4.2) cannot load another session in the same worker, because it queues all session loads
  // on one promise chain and a rejected chain stays rejected.
  prepare(onProgress?: (p: ProgressInfo) => void): Promise<void> {
    return new Promise((resolve, reject) => {
      let attempt = 0
      const listener = (msg: WorkerResponse) => {
        if (msg.type === 'progress') onProgress?.(msg.info)
        if (msg.type === 'ready') { this.device = msg.device; this.attentions = msg.attentions; this.knockout = msg.knockout; this.revision = msg.revision ?? null; this.listeners.delete(listener); resolve() }
        if (msg.type === 'attempt-failed') {
          if (!msg.more) { this.listeners.delete(listener); reject(new Error(msg.message)); return }
          this.worker.terminate()
          this.worker = this.spawn()
          attempt += 1
          this.post({ type: 'prepare', modelId: MODEL_ID, attempt })
        }
        if (msg.type === 'fatal') { this.listeners.delete(listener); reject(new Error(msg.message)) }
      }
      this.listeners.add(listener)
      this.post({ type: 'prepare', modelId: MODEL_ID, attempt })
    })
  }

  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void, format: PromptFormat = TEXT_FORMAT, options?: RunOptions): RunHandle {
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
      this.post({ type: 'run', runId, prompt, params, format, ...(options?.seed !== undefined ? { seed: options.seed } : {}), ...(options?.knockout ? { knockout: options.knockout } : {}) })
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

  sweep(req: Omit<Extract<WorkerRequest, { type: 'sweep-request' }>, 'type' | 'sweepId'>,
    onBaseline: (p: number) => void, onLayer: (e: KnockoutLayer) => void): { abort(): void; done: Promise<SweepResult> } {
    const sweepId = this.nextSweepId++
    const done = new Promise<SweepResult>((resolve) => {
      const listener = (msg: WorkerResponse) => {
        if (msg.type === 'sweep-baseline' && msg.sweepId === sweepId) onBaseline(msg.prob)
        if (msg.type === 'sweep-layer' && msg.sweepId === sweepId) onLayer(msg.entry)
        if (msg.type === 'sweep-done' && msg.sweepId === sweepId) { this.listeners.delete(listener); resolve(msg.error !== undefined ? { ok: false, error: msg.error } : { ok: !msg.aborted }) }
        if (msg.type === 'fatal') { this.listeners.delete(listener); resolve({ ok: false, error: msg.message }) }
      }
      this.listeners.add(listener)
      this.post({ type: 'sweep-request', sweepId, ...req })
    })
    return { abort: () => this.post({ type: 'abort' }), done }
  }
}

/// <reference lib="webworker" />
import type { GenParams, TokenInfo, TraceEvent } from '../../trace/types'
import { sampleIndex, softmax, topK } from '../math'
import type { WorkerRequest, WorkerResponse } from './protocol'

const post = (msg: WorkerResponse) => (self as unknown as Worker).postMessage(msg)

/* eslint-disable @typescript-eslint/no-explicit-any */
let tokenizer: any = null
let model: any = null
let loadedModelId = 'unknown'
let aborted = false

async function prepare(modelId: string) {
  loadedModelId = modelId
  const { AutoTokenizer, AutoModelForCausalLM } = await import('@huggingface/transformers')
  const progress_callback = (p: any) => {
    if (p.status === 'progress') post({ type: 'progress', info: { file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0 } })
  }
  tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback })
  const device = 'gpu' in navigator ? 'webgpu' : 'wasm'
  try {
    model = await AutoModelForCausalLM.from_pretrained(modelId, { dtype: 'q4', device, progress_callback })
    post({ type: 'ready', device })
  } catch {
    model = await AutoModelForCausalLM.from_pretrained(modelId, { dtype: 'q4', device: 'wasm', progress_callback })
    post({ type: 'ready', device: 'wasm' })
  }
}

const tokenInfo = (id: number): TokenInfo => ({ id, text: tokenizer.decode([id]) })

// transformers.js does not export its internal `getPastKeyValues` helper (models/modeling_utils.js),
// so we replicate its rename step (present.* -> past_key_values.*) on top of the public `DynamicCache`
// class to thread the KV cache between generation cycles ourselves.
function updateCache(DynamicCacheCtor: any, output: any, prevCache: any): any {
  const cache = prevCache ?? new DynamicCacheCtor()
  const entries: Record<string, unknown> = {}
  for (const name in output) {
    if (name.startsWith('present')) {
      const newName = name
        .replace('present_ssm', 'past_ssm')
        .replace('present_conv', 'past_conv')
        .replace('present_recurrent', 'past_recurrent')
        .replace('present', 'past_key_values')
      entries[newName] = output[name]
    }
  }
  cache.update(entries)
  return cache
}

async function run(runId: number, prompt: string, params: GenParams) {
  aborted = false
  const emit = (event: TraceEvent) => post({ type: 'trace', runId, event })
  const { Tensor, DynamicCache } = await import('@huggingface/transformers')

  emit({ type: 'run-start', prompt, mode: 'real', modelId: loadedModelId, params,
    ...(model.config.vocab_size ? { vocabSize: model.config.vocab_size } : {}) })
  let promptIds: number[] = tokenizer.encode(prompt, { add_special_tokens: false })
  const maxCtx: number = model.config.max_position_embeddings ?? 2048
  const budget = Math.max(1, maxCtx - params.maxNewTokens)
  const truncated = promptIds.length > budget
  if (truncated) promptIds = promptIds.slice(-budget)   // keep the most recent tokens
  emit({ type: 'tokenize', tokens: promptIds.map(tokenInfo), ...(truncated ? { truncated } : {}) })

  const numLayers: number = model.config.num_hidden_layers ?? 12
  const dims: number = model.config.hidden_size ?? 576
  // eos_token_id may be a single id or an array of ids; prefer the model's generation_config,
  // fall back to raw config, then the tokenizer (matching engine/tokenizer.ts's lookup order).
  const rawEos = model.generation_config?.eos_token_id
    ?? model.config.eos_token_id
    ?? tokenizer.model?.eos_token_id
    ?? tokenizer.eos_token_id
    ?? -1
  const eosIds: number[] = Array.isArray(rawEos) ? rawEos : [rawEos]
  const allIds = [...promptIds]
  let pastKeyValues: any = null
  let nextInputIds = promptIds

  try {
    for (let cycle = 0; cycle < params.maxNewTokens; cycle++) {
      if (aborted) { emit({ type: 'run-end', reason: 'aborted' }); break }

      // schematic embed preview (real hidden states not exposed; spec-accepted compromise)
      emit({ type: 'embed', cycle, seqLen: allIds.length, dims,
        preview: allIds.slice(-4).map((id) => Array.from({ length: 16 }, (_, d) => Math.sin(id * 0.7 + d))) })
      for (let l = 0; l < numLayers; l++) emit({ type: 'layer', cycle, index: l, total: numLayers })

      const input_ids = new Tensor('int64', BigInt64Array.from(nextInputIds.map(BigInt)), [1, nextInputIds.length])
      const attention_mask = new Tensor('int64', BigInt64Array.from(allIds.map(() => 1n)), [1, allIds.length])
      const out = await model({ input_ids, attention_mask, past_key_values: pastKeyValues })
      pastKeyValues = updateCache(DynamicCache, out, pastKeyValues)

      const [, seq, vocab] = out.logits.dims as [number, number, number]
      const lastLogits: Float32Array = out.logits.data.slice((seq - 1) * vocab, seq * vocab)
      const top = topK(lastLogits, params.topK).map((c) => ({ ...tokenInfo(c.id), logit: Math.round(c.logit * 100) / 100 }))
      emit({ type: 'logits', cycle, topK: top })

      const probs = softmax(top.map((c) => c.logit), params.temperature)
      emit({ type: 'softmax', cycle, temperature: params.temperature,
        topK: top.map((c, i) => ({ id: c.id, text: c.text, prob: probs[i] })) })

      const method = params.temperature === 0 ? 'greedy' : 'top-k'
      const idx = method === 'greedy' ? 0 : sampleIndex(probs, Math.random)
      const chosen = { id: top[idx].id, text: top[idx].text }
      emit({ type: 'sample', cycle, chosen, method })
      emit({ type: 'append', cycle, token: chosen })

      allIds.push(chosen.id)
      nextInputIds = [chosen.id]
      if (eosIds.includes(chosen.id)) { emit({ type: 'run-end', reason: 'eos' }); break }
      if (cycle === params.maxNewTokens - 1) emit({ type: 'run-end', reason: 'max-tokens' })
    }
  } finally {
    // Every exit path (eos/max-tokens/aborted break, or an exception) must release the final
    // cycle's cache tensors — DynamicCache.update() only disposes *replaced* GPU-buffer tensors,
    // so the last cycle's tensors are never freed unless we dispose the cache here (mirrors the
    // library's own generate(), which does `await past_key_values.dispose()` after its loop).
    await (pastKeyValues as any)?.dispose?.()
  }
  post({ type: 'done', runId })
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  try {
    const msg = e.data
    if (msg.type === 'prepare') await prepare(msg.modelId)
    if (msg.type === 'run') await run(msg.runId, msg.prompt, msg.params)
    if (msg.type === 'abort') aborted = true
  } catch (err) {
    post({ type: 'fatal', message: err instanceof Error ? err.message : String(err) })
  }
}

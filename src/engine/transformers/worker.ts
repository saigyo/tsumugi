/// <reference lib="webworker" />
import type { GenParams, RunEndReason, TokenInfo, TraceEvent } from '../../trace/types'
import { ATTN_MODEL_ID } from '../tokenizer'
import { sampleIndex, softmax, topK } from '../math'
import { addAttentionOutput, createAccumulator, type AttnAccumulator } from './attentionAccum'
import type { WorkerRequest, WorkerResponse } from './protocol'
import { headStats, resolveHeadLabel, selectShowcaseHeads, type HeadStats, type ShowcasePrev } from './attentionStats'
import { buildGridCells } from './attentionThumbs'

const post = (msg: WorkerResponse) => (self as unknown as Worker).postMessage(msg)

// Attention accumulation is O(seq^2) in both matrix storage (ragged per-layer/per-head
// rows) and per-cycle stats cost (headStats/selectShowcaseHeads rescan every row every
// cycle); at ~2000 tokens across 270 attention outputs this balloons into multi-GB. Cap
// the combined prompt+generation length we're willing to accumulate for and fall back to
// schematic (acc stays null) above it — generation itself is unaffected.
const ATTN_MAX_SEQ = 512

/* eslint-disable @typescript-eslint/no-explicit-any */
let tokenizer: any = null
let model: any = null
let loadedModelId = 'unknown'
let hasAttentions = false
let aborted = false
// finished run kept for the head-request side channel until the next run
// starts (memory already bounded by ATTN_MAX_SEQ)
let lastRun: { acc: AttnAccumulator; stats: HeadStats[] } | null = null

async function prepare(modelId: string) {
  loadedModelId = modelId
  const { AutoTokenizer, AutoModelForCausalLM } = await import('@huggingface/transformers')
  const progress_callback = (p: any) => {
    if (p.status === 'progress') post({ type: 'progress', info: { file: p.file, loaded: p.loaded ?? 0, total: p.total ?? 0 } })
  }
  tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback })
  const preferred = 'gpu' in navigator ? 'webgpu' : 'wasm'

  // WebGPU->WASM fallback for a single (modelId, dtype) attempt; returns the device that
  // actually succeeded, since a caught GPU failure means the load happened on WASM instead.
  const loadOn = async (id: string, dtype: 'q4' | 'q8' | 'fp16'): Promise<{ m: any; device: 'webgpu' | 'wasm' }> => {
    try {
      return { m: await AutoModelForCausalLM.from_pretrained(id, { dtype, device: preferred, progress_callback }), device: preferred }
    } catch {
      return { m: await AutoModelForCausalLM.from_pretrained(id, { dtype, device: 'wasm', progress_callback }), device: 'wasm' }
    }
  }

  let device: 'webgpu' | 'wasm'
  try {
    // q8 is the attn repo's sole published variant: greedy-token-identical to
    // fp32 (lm_head excluded from quantization). q4/fp16 were disqualified
    // during validation — see tools/export/src/tsumugi_export/quantize.py.
    ;({ m: model, device } = await loadOn(ATTN_MODEL_ID, 'q8'))
    loadedModelId = ATTN_MODEL_ID
    hasAttentions = true
  } catch {
    try {
      ;({ m: model, device } = await loadOn(ATTN_MODEL_ID, 'fp16'))
      loadedModelId = ATTN_MODEL_ID
      hasAttentions = true
    } catch {
      ;({ m: model, device } = await loadOn(modelId, 'q4'))
      loadedModelId = modelId
      hasAttentions = false
    }
  }
  post({ type: 'ready', device, attentions: hasAttentions })
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
  lastRun = null
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

  const numHeads: number = model.config.num_attention_heads ?? 9
  const acc = hasAttentions && promptIds.length + params.maxNewTokens <= ATTN_MAX_SEQ
    ? createAccumulator(numLayers, numHeads)
    : null
  let attnBroken = false
  let stats: HeadStats[] | null = null
  let prevSel: ShowcasePrev = {}

  // Grid emission shares the never-fail policy: a failure flips attnBroken
  // and the run still ends normally, just without a grid.
  const endRun = (reason: RunEndReason) => {
    if (acc && !attnBroken && stats) {
      try {
        emit({ type: 'attention-grid', layers: acc.layers, heads: acc.heads,
          cells: buildGridCells(acc, stats) })
        lastRun = { acc, stats }
      } catch { attnBroken = true }
    }
    emit({ type: 'run-end', reason })
  }

  try {
    for (let cycle = 0; cycle < params.maxNewTokens; cycle++) {
      if (aborted) { endRun('aborted'); break }

      // schematic embed preview (real hidden states not exposed; spec-accepted compromise)
      emit({ type: 'embed', cycle, seqLen: allIds.length, dims,
        preview: allIds.slice(-4).map((id) => Array.from({ length: 16 }, (_, d) => Math.sin(id * 0.7 + d))) })
      for (let l = 0; l < numLayers; l++) emit({ type: 'layer', cycle, index: l, total: numLayers })

      const input_ids = new Tensor('int64', BigInt64Array.from(nextInputIds.map(BigInt)), [1, nextInputIds.length])
      const attention_mask = new Tensor('int64', BigInt64Array.from(allIds.map(() => 1n)), [1, allIds.length])
      const out = await model({ input_ids, attention_mask, past_key_values: pastKeyValues })
      pastKeyValues = updateCache(DynamicCache, out, pastKeyValues)

      if (acc && !attnBroken) {
        try {
          for (let l = 0; l < numLayers; l++) {
            const t = out[`attentions.${l}`]
            if (!t) throw new Error(`missing attentions.${l}`)
            addAttentionOutput(acc, l, t.dims as number[], t.data as Float32Array)
          }
        } catch { attnBroken = true }
      }

      // Emit before any further addAttentionOutput call: selectShowcaseHeads' matrices are live
      // references into the accumulator, and postMessage only clones them synchronously here.
      // Never let a stats/emit failure escalate to fatal (kills generation) — degrade to
      // schematic instead, same as the accumulation try/catch above.
      if (acc && !attnBroken) {
        try {
          stats = headStats(acc, allIds.map(tokenInfo))
          const heads = selectShowcaseHeads(stats, acc, 0.3, prevSel)
          prevSel = Object.fromEntries(heads.map((h) => [h.label, { layer: h.layer, head: h.head }]))
          if (heads.length > 0) emit({ type: 'attention', cycle, heads })
        } catch { attnBroken = true }
      }

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
      if (eosIds.includes(chosen.id)) { endRun('eos'); break }
      if (cycle === params.maxNewTokens - 1) endRun('max-tokens')
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
    if (msg.type === 'head-request') {
      const r = lastRun
      const ok = r !== null
        && msg.layer >= 0 && msg.layer < r.acc.layers
        && msg.head >= 0 && msg.head < r.acc.heads
      const resolved = ok ? resolveHeadLabel(r.stats, msg.layer, msg.head) : { label: null, score: null }
      post({ type: 'head-response', layer: msg.layer, head: msg.head,
        matrix: ok ? r.acc.rows[msg.layer][msg.head] : [],
        label: resolved.label, score: resolved.score })
    }
  } catch (err) {
    post({ type: 'fatal', message: err instanceof Error ? err.message : String(err) })
  }
}

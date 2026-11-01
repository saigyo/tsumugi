/// <reference lib="webworker" />
import type { ChatSpans, GenParams, KnockoutLayer, LayerVotes, NeuronData, PromptFormat, RunEndReason, TokenInfo, TraceEvent } from '../../trace/types'
import { chatSpans, fitChatToBudget, noSystemChat, TEXT_FORMAT } from '../chatTemplate'
import { forcedToken, mlpScale, sweepEntry } from '../knockout'
import { cycleDraw } from '../prng'
import { sampleIndex, softmax, topK } from '../math'
import { ATTN_MODEL_ID, ATTN_MODEL_REVISION } from '../tokenizer'
import { addAttentionOutput, createAccumulator, type AttnAccumulator } from './attentionAccum'
import { extractEmbedRows } from './embedRows'
import { extractReadout, type ReadoutResult } from './readout'
import { computeVotes, extractEdits } from './mlpData'
import { extractNeurons, extractSteps } from './neuronData'
import type { RunOptions } from '../types'
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

// Real embedding rows are emitted in full (dims floats per token) on every cycle; at
// ~2000 prompt tokens that's ~1.15M numbers in one postMessage and ~7 MB per archived
// run. Cap the prompt length we're willing to emit real rows for and fall back to
// 'asset' rows above it — generation itself is unaffected.
const EMBED_MAX_PROMPT = 512

/* eslint-disable @typescript-eslint/no-explicit-any */
let tokenizer: any = null
let model: any = null
let loadedModelId = 'unknown'
let hasAttentions = false
// v7 export: the graph has the required mlp_scale input (absent on v6 sessions)
let hasScale = false
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
  const loadOn = async (id: string, dtype: 'q4' | 'q8' | 'fp16', revision = 'main'): Promise<{ m: any; device: 'webgpu' | 'wasm' }> => {
    const opts = { dtype, progress_callback, revision }
    try {
      return { m: await AutoModelForCausalLM.from_pretrained(id, { ...opts, device: preferred }), device: preferred }
    } catch {
      return { m: await AutoModelForCausalLM.from_pretrained(id, { ...opts, device: 'wasm' }), device: 'wasm' }
    }
  }

  let device: 'webgpu' | 'wasm'
  try {
    // model_quantized.onnx (dtype 'q8') is the attn repo's sole published variant: from v6
    // weight-only 8-bit with fp32 activations, the same next token as fp32 in browsers (#19);
    // see tools/export/src/tsumugi_export/quantize.py.
    ;({ m: model, device } = await loadOn(ATTN_MODEL_ID, 'q8', ATTN_MODEL_REVISION))
    loadedModelId = ATTN_MODEL_ID
    hasAttentions = true
  } catch {
    try {
      ;({ m: model, device } = await loadOn(ATTN_MODEL_ID, 'fp16', ATTN_MODEL_REVISION))
      loadedModelId = ATTN_MODEL_ID
      hasAttentions = true
    } catch {
      ;({ m: model, device } = await loadOn(modelId, 'q4'))
      loadedModelId = modelId
      hasAttentions = false
    }
  }
  // Guarded like the other export features: a missing sessions/inputNames means no knockout.
  try { hasScale = model.sessions.model.inputNames.includes('mlp_scale') } catch { hasScale = false }
  post({ type: 'ready', device, attentions: hasAttentions, knockout: hasScale,
    ...(loadedModelId === ATTN_MODEL_ID ? { revision: ATTN_MODEL_REVISION } : {}) })
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

async function run(runId: number, prompt: string, params: GenParams, format: PromptFormat, knockout?: RunOptions['knockout']) {
  aborted = false
  lastRun = null
  const emit = (event: TraceEvent) => post({ type: 'trace', runId, event })
  const { Tensor, DynamicCache } = await import('@huggingface/transformers')

  // a knockout run on a session that cannot remove a layer would be recorded as a knockout
  // while running unmodified: refuse it
  if (knockout && !hasScale) {
    emit({ type: 'run-end', reason: 'error', message: 'This model cannot remove a layer’s MLP, so the knockout run was not started.' })
    post({ type: 'done', runId })
    return
  }

  const seed = knockout?.seed ?? crypto.getRandomValues(new Uint32Array(1))[0]
  emit({ type: 'run-start', prompt, mode: 'real', modelId: loadedModelId, params, seed,
    ...(loadedModelId === ATTN_MODEL_ID ? { revision: ATTN_MODEL_REVISION } : {}),
    ...(format.kind === 'chat' ? { format } : {}),
    ...(knockout ? { knockout: knockout.origin } : {}),
    ...(model.config.vocab_size ? { vocabSize: model.config.vocab_size } : {}) })
  const maxCtx: number = model.config.max_position_embeddings ?? 2048
  const budget = Math.max(1, maxCtx - params.maxNewTokens)
  let promptIds: number[]
  let chat: ChatSpans | undefined
  let truncated = false
  if (format.kind === 'chat') {
    const text: string = format.system === null
      ? noSystemChat(prompt)
      : tokenizer.apply_chat_template(
        [{ role: 'system', content: format.system }, { role: 'user', content: prompt }],
        { tokenize: false, add_generation_prompt: true })
    promptIds = tokenizer.encode(text, { add_special_tokens: false })
    chat = chatSpans(promptIds, {
      start: tokenizer.convert_tokens_to_ids('<|im_start|>'), end: tokenizer.convert_tokens_to_ids('<|im_end|>'),
      newline: tokenizer.encode('\n', { add_special_tokens: false })[0],
    }, format.system !== null)
    if (chat) {
      ;({ ids: promptIds, spans: chat, truncated } = fitChatToBudget(promptIds, chat, budget))
    }
  } else {
    promptIds = tokenizer.encode(prompt, { add_special_tokens: false })
  }
  if (promptIds.length > budget) { promptIds = promptIds.slice(-budget); truncated = true; chat = undefined }
  emit({ type: 'tokenize', tokens: promptIds.map(tokenInfo),
    ...(truncated ? { truncated } : {}), ...(chat ? { chat } : {}) })

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
  // Off for the whole run when the prompt exceeds EMBED_MAX_PROMPT (source stays
  // 'asset' throughout), or flipped mid-run by a bad-shape inputs_embeds below —
  // never-fail, like attnBroken.
  let embedsOff = promptIds.length > EMBED_MAX_PROMPT
  // Readout for the Logits card (v3 export); flipped by a bad shape — never-fail.
  let readoutOff = false
  // MLP cards (v4 export): each flipped by a bad shape — never-fail.
  let editsOff = false
  let votesOff = false
  // neuron card (v5 export): flipped by a bad shape — never-fail
  let neuronsOff = false
  // step-by-step card (v6 export): flipped by a bad shape — never-fail, the neuron card keeps working
  let stepsOff = false
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

      const input_ids = new Tensor('int64', BigInt64Array.from(nextInputIds.map(BigInt)), [1, nextInputIds.length])
      const attention_mask = new Tensor('int64', BigInt64Array.from(allIds.map(() => 1n)), [1, allIds.length])
      const inputs: Record<string, unknown> = { input_ids, attention_mask, past_key_values: pastKeyValues }
      if (hasScale) inputs.mlp_scale = new Tensor('float32', mlpScale(numLayers, knockout?.origin.layer ?? null), [numLayers])
      const out = await model(inputs)
      pastKeyValues = updateCache(DynamicCache, out, pastKeyValues)

      // Real embedding rows for the tokens fed this cycle (whole prompt at cycle 0,
      // one token afterwards). Absent output = old cached export → 'asset' quietly;
      // a wrong shape flips embedsOff for the run — never-fail, like attention.
      let rows: number[][] | undefined
      if (!embedsOff) {
        const r = extractEmbedRows(out.inputs_embeds, nextInputIds.length, dims)
        if (r.status === 'ok') rows = r.rows
        else if (r.status === 'bad-shape') {
          embedsOff = true
          console.warn(`inputs_embeds has shape [${r.dims.join(', ')}], expected [1, ${nextInputIds.length}, ${dims}] — using asset vectors`)
        }
      }
      emit(rows
        ? { type: 'embed', cycle, seqLen: allIds.length, dims, source: 'model', rows }
        : { type: 'embed', cycle, seqLen: allIds.length, dims, source: 'asset' })
      for (let l = 0; l < numLayers; l++) emit({ type: 'layer', cycle, index: l, total: numLayers })

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
      // After `top`: the neuron data carries the rows of exactly these candidates.
      if (!editsOff) {
        const r = extractEdits(out, nextInputIds.length, numLayers)
        if (r.status === 'ok') {
          let neurons: NeuronData | undefined
          if (!neuronsOff) {
            const n = extractNeurons(out, numLayers, dims, top.map((c) => c.id))
            if (n.status === 'ok') {
              neurons = n.data
              if (!stepsOff) {
                const s = extractSteps(out, numLayers, n.data.neurons, dims)
                if (s.status === 'ok') neurons = { ...n.data, steps: s.steps }
                else if (s.status === 'bad-shape') {
                  stepsOff = true
                  console.warn(`step outputs unusable (${s.detail}) — the step-by-step card shows its note`)
                }
              }
            } else if (n.status === 'bad-shape') {
              neuronsOff = true
              console.warn(`neuron outputs unusable (${n.detail}) — the neuron card shows its note`)
            }
          }
          emit({ type: 'edits', cycle, start: allIds.length - nextInputIds.length, attn: r.attn, mlp: r.mlp,
            ...(neurons ? { neurons } : {}) })
        } else if (r.status === 'bad-shape') {
          editsOff = true
          console.warn(`edit_ratios unusable (${r.detail}) — the edit-size card shows its note`)
        }
      }

      // Readout for the Logits card (v3 export): never-fail, like inputs_embeds.
      let readout: ReadoutResult = { status: 'absent' }
      if (!readoutOff) {
        readout = extractReadout(out, nextInputIds.length, dims)
        if (readout.status === 'bad-shape') {
          readoutOff = true
          console.warn(`readout outputs unusable (${readout.detail}) — the Logits card shows the schematic readout`)
        }
      }
      let votes = new Map<number, LayerVotes>()
      if (!votesOff) {
        const v = computeVotes(out, numLayers, dims, (id) => lastLogits[id])
        if (v.status === 'ok') votes = v.byId
        else if (v.status === 'bad-shape') {
          votesOff = true
          console.warn(`stream_parts unusable (${v.detail}) — the votes card shows its note`)
        }
      }
      const candidates = top.map((c) => {
        const r = readout.status === 'ok' ? readout.byId.get(c.id) : undefined
        const v = votes.get(c.id)
        return { ...c, ...(r ? { row: r.row, dot: r.dot } : {}), ...(v ? { votes: v } : {}) }
      })
      emit({ type: 'logits', cycle, topK: candidates,
        ...(readout.status === 'ok' ? { lastHidden: readout.lastHidden } : {}) })

      const probs = softmax(top.map((c) => c.logit), params.temperature)
      emit({ type: 'softmax', cycle, temperature: params.temperature,
        topK: top.map((c, i) => ({ id: c.id, text: c.text, prob: probs[i] })) })

      // Knockout runs replay the original run's tokens for the copied prefix.
      const forcedId = knockout ? forcedToken(knockout.forced, cycle) : null
      const method = forcedId !== null ? 'forced' : params.temperature === 0 ? 'greedy' : 'top-k'
      const idx = method === 'top-k' ? sampleIndex(probs, () => cycleDraw(seed, cycle)) : 0
      const chosen = forcedId !== null
        ? tokenInfo(forcedId)
        : { id: top[idx].id, text: top[idx].text }
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

// Knockout sweep: one baseline pass, then one pass per layer with that layer's MLP removed,
// each over the full `ids` without the KV cache. Never-fail: a session without mlp_scale
// answers aborted.
async function sweep(msg: Extract<WorkerRequest, { type: 'sweep-request' }>) {
  aborted = false
  const { sweepId } = msg
  if (!hasScale) { post({ type: 'sweep-done', sweepId, aborted: true }); return }
  const { Tensor } = await import('@huggingface/transformers')
  const numLayers: number = model.config.num_hidden_layers ?? 12
  const draw = cycleDraw(msg.seed, msg.cycle)
  const n = msg.ids.length
  const input_ids = new Tensor('int64', BigInt64Array.from(msg.ids.map(BigInt)), [1, n])
  const attention_mask = new Tensor('int64', BigInt64Array.from(msg.ids.map(() => 1n)), [1, n])
  let stopped = false
  try {
    // off = -1 is the baseline (all ones)
    for (let off = -1; off < numLayers; off++) {
      if (aborted) { stopped = true; break }
      const mlp_scale = new Tensor('float32', mlpScale(numLayers, off < 0 ? null : off), [numLayers])
      const out = await model({ input_ids, attention_mask, mlp_scale })
      try {
        const [, seq, vocab] = out.logits.dims as [number, number, number]
        const last: Float32Array = out.logits.data.slice((seq - 1) * vocab, seq * vocab)
        const e = sweepEntry(last, msg.params, msg.chosenId, draw)
        if (off < 0) post({ type: 'sweep-baseline', sweepId, prob: e.prob })
        else {
          const entry: KnockoutLayer = { layer: off, prob: e.prob,
            top: { ...tokenInfo(e.topId), prob: e.topProb }, flips: e.flips }
          post({ type: 'sweep-layer', sweepId, entry })
        }
      } finally {
        // free this pass's output tensors (GPU buffers) so 31 passes do not accumulate
        for (const k in out) (out[k] as any)?.dispose?.()
      }
    }
  } catch (err) {
    post({ type: 'sweep-done', sweepId, aborted: true, error: err instanceof Error ? err.message : String(err) })
    return
  }
  post({ type: 'sweep-done', sweepId, aborted: stopped })
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  try {
    const msg = e.data
    if (msg.type === 'prepare') await prepare(msg.modelId)
    if (msg.type === 'run') await run(msg.runId, msg.prompt, msg.params, msg.format ?? TEXT_FORMAT, msg.knockout)
    if (msg.type === 'abort') aborted = true
    if (msg.type === 'sweep-request') await sweep(msg)
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

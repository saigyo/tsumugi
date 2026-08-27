import type { GenParams, TokenInfo, TraceEvent } from '../../trace/types'
import { sampleIndex, softmax } from '../math'
import { mulberry32, seedFromTokens } from '../prng'
import type { PipelineEngine, RunHandle } from '../types'
import type { Tokenizer } from '../tokenizer'
import { MODEL_ID } from '../tokenizer'
import { candidateWords } from './candidates'

export class SimulatedEngine implements PipelineEngine {
  private tokenizer: Tokenizer
  private layers: number
  private dims: number

  constructor(tokenizer: Tokenizer, opts?: { layers?: number; dims?: number }) {
    this.tokenizer = tokenizer
    this.layers = opts?.layers ?? 12
    this.dims = opts?.dims ?? 576
  }

  async prepare(): Promise<void> {}

  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void): RunHandle {
    let aborted = false
    const done = this.loop(prompt, params, emit, () => aborted)
    return { abort: () => { aborted = true }, done }
  }

  private async loop(
    prompt: string, params: GenParams,
    emit: (e: TraceEvent) => void, isAborted: () => boolean,
  ): Promise<void> {
    emit({ type: 'run-start', prompt, mode: 'sim', modelId: MODEL_ID, params })
    const promptTokens = this.tokenizer.encode(prompt)
    emit({ type: 'tokenize', tokens: promptTokens })

    const rand = mulberry32(seedFromTokens(promptTokens.map((t) => t.id)))
    const seq: TokenInfo[] = [...promptTokens]
    let text = prompt

    for (let cycle = 0; cycle < params.maxNewTokens; cycle++) {
      await new Promise((r) => setTimeout(r, 0))
      if (isAborted()) { emit({ type: 'run-end', reason: 'aborted' }); return }

      const preview = seq.slice(-4).map(() =>
        Array.from({ length: 16 }, () => Math.round((rand() * 2 - 1) * 100) / 100))
      emit({ type: 'embed', cycle, seqLen: seq.length, dims: this.dims, preview })

      for (let i = 0; i < this.layers; i++)
        emit({ type: 'layer', cycle, index: i, total: this.layers,
          activationNorm: Math.round((1 + 0.05 * i + 0.3 * rand()) * 100) / 100 })

      const k = Math.min(10, Math.max(1, params.topK))
      const candidates = candidateWords(text, rand).map((word) => {
        const tok = this.tokenizer.encode(word)[0]
        return { id: tok.id, text: tok.text }
      })
      const scored = candidates
        .map((c, i) => ({ ...c, logit: 10 - i * 1.1 + rand() * 0.6 }))
        .sort((a, b) => b.logit - a.logit)
        .slice(0, k)
      emit({ type: 'logits', cycle, topK: scored })

      const probs = softmax(scored.map((c) => c.logit), params.temperature)
      emit({ type: 'softmax', cycle, temperature: params.temperature,
        topK: scored.map((c, i) => ({ id: c.id, text: c.text, prob: probs[i] })) })

      const method = params.temperature === 0 ? 'greedy' : 'top-k'
      const idx = method === 'greedy' ? 0 : sampleIndex(probs, rand)
      const chosen = { id: scored[idx].id, text: scored[idx].text }
      emit({ type: 'sample', cycle, chosen, method })
      emit({ type: 'append', cycle, token: chosen })
      seq.push(chosen)
      text += chosen.text

      if (chosen.text.trim() === '.' && rand() < 0.03 * cycle) {
        emit({ type: 'run-end', reason: 'eos' })
        return
      }
    }
    emit({ type: 'run-end', reason: 'max-tokens' })
  }
}

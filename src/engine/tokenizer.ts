import type { TokenInfo } from '../trace/types'
import { setTokenPieces } from './tokenBytes'

export interface Tokenizer {
  encode(text: string): TokenInfo[]
  eosTokenId: number
}

export const MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'
export const ATTN_MODEL_ID = 'saigyo-hoshi/smollm2-135m-attn-onnx'
// Hub tag the app loads the model and the geometry asset from. Transformers.js
// caches files by URL and never revalidates, so a new release needs a new
// tag here to reach browsers that cached the previous one. v3 = readout
// outputs (last_hidden_state, topk_ids, topk_embeds). v4 = MLP internals
// (edit_ratios, stream_parts). v5 = neuron outputs (mlp_activations,
// final_norm_inv_rms) and the neurons/ asset. v6 = weight-only 8-bit model
// (#19) and the step outputs (mlp_gate, mlp_up, mlp_in, mlp_out); the neurons
// manifest names its revision.
export const ATTN_MODEL_REVISION = 'v7'

// Both ids share the same weights, so one geometry asset covers both.
export const GEOMETRY_MODEL_IDS: readonly string[] = [MODEL_ID, ATTN_MODEL_ID]
export const geometryUrlFor = (modelId: string, revision: string) =>
  `https://huggingface.co/${modelId}/resolve/${revision}/geometry`
export const GEOMETRY_BASE_URL: string =
  import.meta.env.VITE_GEOMETRY_BASE_URL ?? geometryUrlFor(ATTN_MODEL_ID, ATTN_MODEL_REVISION)
export const neuronsUrlFor = (modelId: string, revision: string) =>
  `https://huggingface.co/${modelId}/resolve/${revision}/neurons`
export const NEURONS_BASE_URL: string =
  import.meta.env.VITE_NEURONS_BASE_URL ?? neuronsUrlFor(ATTN_MODEL_ID, ATTN_MODEL_REVISION)

function hashId(text: string): number {
  let h = 0x811c9dc5
  for (const ch of text) {
    h ^= ch.codePointAt(0) ?? 0
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0) % 50000
}

export function fallbackTokenizer(): Tokenizer {
  return {
    eosTokenId: 0,
    encode: (text) => (text.match(/\s*\S+/g) ?? []).map((chunk) => ({ id: hashId(chunk), text: chunk })),
  }
}

export const fakeTokenizer = fallbackTokenizer

export async function loadTokenizer(modelId: string = MODEL_ID): Promise<Tokenizer> {
  try {
    const { AutoTokenizer } = await import('@huggingface/transformers')
    const tok = await AutoTokenizer.from_pretrained(modelId)
    /* eslint-disable @typescript-eslint/no-explicit-any */
    const eos: number = (tok as any).model?.eos_token_id ?? (tok as any).eos_token_id ?? 0
    // raw vocabulary pieces, so labels of partial-character tokens can show their bytes
    let pieces: string[] | null = null
    setTokenPieces((id) => {
      if (!pieces) {
        pieces = []
        const vocab = (tok as any).get_vocab() as Map<string, number> | Record<string, number>
        for (const [piece, i] of vocab instanceof Map ? vocab : Object.entries(vocab)) pieces[i] = piece
      }
      return pieces[id]
    })
    return {
      eosTokenId: eos,
      encode: (text) => {
        const ids: number[] = tok.encode(text, { add_special_tokens: false } as any)
        return ids.map((id) => ({ id, text: tok.decode([id]) }))
      },
    }
  } catch {
    return fallbackTokenizer()
  }
}

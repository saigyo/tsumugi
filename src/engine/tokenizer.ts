import type { TokenInfo } from '../trace/types'

export interface Tokenizer {
  encode(text: string): TokenInfo[]
  eosTokenId: number
}

export const MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'
export const ATTN_MODEL_ID = 'saigyo-hoshi/smollm2-135m-attn-onnx'
// Hub tag the app loads the model and the geometry asset from. Transformers.js
// caches files by URL and never revalidates, so a new release needs a new
// tag here to reach browsers that cached the previous one. v3 = readout
// outputs (last_hidden_state, topk_ids, topk_embeds).
export const ATTN_MODEL_REVISION = 'v3'

// Both ids share the same weights, so one geometry asset covers both.
export const GEOMETRY_MODEL_IDS: readonly string[] = [MODEL_ID, ATTN_MODEL_ID]
export const geometryUrlFor = (modelId: string, revision: string) =>
  `https://huggingface.co/${modelId}/resolve/${revision}/geometry`
export const GEOMETRY_BASE_URL: string =
  import.meta.env.VITE_GEOMETRY_BASE_URL ?? geometryUrlFor(ATTN_MODEL_ID, ATTN_MODEL_REVISION)

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

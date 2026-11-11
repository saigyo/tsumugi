import { ATTN_MODEL_ID, ATTN_MODEL_REVISION } from '../tokenizer'

export type Device = 'webgpu' | 'wasm'

/** One way of loading the model: which files, and on which device. */
export interface LoadAttempt {
  modelId: string
  dtype: 'q8' | 'fp16' | 'q4'
  revision: string
  device: Device
  // the custom export with attention and readout outputs (the stock export has none)
  attentions: boolean
}

/**
 * The ways of loading the model, best first. The custom export comes before the stock one, and for each of
 * them the graphics card before the CPU. WebGPU is only in the list when an adapter can really be had: a
 * browser may offer `navigator.gpu` and still have none.
 *
 * Each attempt needs a worker of its own (see TransformersEngine.prepare): after one failed session load,
 * transformers.js cannot load another session in the same worker.
 */
export function loadPlan(stockModelId: string, hasAdapter: boolean): LoadAttempt[] {
  const devices: Device[] = hasAdapter ? ['webgpu', 'wasm'] : ['wasm']
  const exports: Array<Omit<LoadAttempt, 'device'>> = [
    // model_quantized.onnx (dtype 'q8') is the attn repo's sole published variant: from v6 weight-only 8-bit
    // with fp32 activations, the same next token as fp32 in browsers (#19); see
    // tools/export/src/tsumugi_export/quantize.py.
    { modelId: ATTN_MODEL_ID, dtype: 'q8', revision: ATTN_MODEL_REVISION, attentions: true },
    { modelId: ATTN_MODEL_ID, dtype: 'fp16', revision: ATTN_MODEL_REVISION, attentions: true },
    { modelId: stockModelId, dtype: 'q4', revision: 'main', attentions: false },
  ]
  return exports.flatMap((e) => devices.map((device) => ({ ...e, device })))
}

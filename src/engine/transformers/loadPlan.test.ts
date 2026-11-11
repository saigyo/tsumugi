import { expect, test } from 'vitest'
import { ATTN_MODEL_ID, ATTN_MODEL_REVISION } from '../tokenizer'
import { loadPlan } from './loadPlan'

const brief = (stock: string, adapter: boolean) =>
  loadPlan(stock, adapter).map((a) => `${a.modelId === stock ? 'stock' : 'attn'} ${a.dtype} ${a.device}`)

test('with a graphics adapter each export is tried on the graphics card first, then on the CPU', () => {
  expect(brief('some/stock-model', true)).toEqual([
    'attn q8 webgpu', 'attn q8 wasm', 'attn fp16 webgpu', 'attn fp16 wasm', 'stock q4 webgpu', 'stock q4 wasm',
  ])
})

test('without an adapter WebGPU is not tried at all, even where the browser offers navigator.gpu', () => {
  expect(brief('some/stock-model', false)).toEqual(['attn q8 wasm', 'attn fp16 wasm', 'stock q4 wasm'])
})

test('the custom export is loaded at its pinned revision and has attention outputs; the stock export has none', () => {
  const [first] = loadPlan('some/stock-model', false)
  expect(first).toMatchObject({ modelId: ATTN_MODEL_ID, revision: ATTN_MODEL_REVISION, attentions: true })
  expect(loadPlan('some/stock-model', false).at(-1)).toMatchObject({ modelId: 'some/stock-model', revision: 'main', attentions: false })
})

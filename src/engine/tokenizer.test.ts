import { expect, test, vi } from 'vitest'
import { ATTN_MODEL_ID, ATTN_MODEL_REVISION, GEOMETRY_BASE_URL, fallbackTokenizer, geometryUrlFor, loadTokenizer, neuronsUrlFor } from './tokenizer'

// hoisted by vitest — must be top-level, affects the whole file
vi.mock('@huggingface/transformers', () => { throw new Error('offline') })

test('fallback splits words keeping leading spaces', () => {
  const toks = fallbackTokenizer().encode('The cat sat.')
  expect(toks.map((t) => t.text)).toEqual(['The', ' cat', ' sat.'])
})

test('fallback ids are stable', () => {
  const t = fallbackTokenizer()
  expect(t.encode('cat')[0].id).toBe(t.encode('cat')[0].id)
})

test('loadTokenizer falls back when HF import fails', async () => {
  const tok = await loadTokenizer()
  expect(tok.encode('hi')[0].text).toBe('hi')
})

test('geometry asset URL is pinned to the model revision', () => {
  expect(geometryUrlFor('a/b', 'v9')).toBe('https://huggingface.co/a/b/resolve/v9/geometry')
  expect(ATTN_MODEL_REVISION).toMatch(/^v\d+$/)
  if (!import.meta.env.VITE_GEOMETRY_BASE_URL)
    expect(GEOMETRY_BASE_URL).toBe(geometryUrlFor(ATTN_MODEL_ID, ATTN_MODEL_REVISION))
})

test('the neuron asset lives next to the model at the pinned tag', () => {
  expect(ATTN_MODEL_REVISION).toBe('v7')
  expect(neuronsUrlFor('a/b', 'v6')).toBe('https://huggingface.co/a/b/resolve/v6/neurons')
})

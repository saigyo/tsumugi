import { expect, test, vi } from 'vitest'
import { fallbackTokenizer, loadTokenizer } from './tokenizer'

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

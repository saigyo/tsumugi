import { expect, test } from 'vitest'
import { fakeTokenizer } from '../tokenizer'
import { CURATED_EXAMPLES, attentionHeadsFor } from './examples'

const tokenize = (text: string) => fakeTokenizer().encode(text)

test('three curated examples with label, prompt, and hint', () => {
  expect(CURATED_EXAMPLES).toHaveLength(3)
  for (const ex of CURATED_EXAMPLES) {
    expect(ex.id.length).toBeGreaterThan(0)
    expect(ex.label.length).toBeGreaterThan(0)
    expect(ex.prompt.length).toBeGreaterThan(0)
    expect(ex.hint.length).toBeGreaterThan(0)
  }
})

test('coreference example yields a coreference head pointing it → cat', () => {
  const ex = CURATED_EXAMPLES.find((e) => e.id === 'coreference')!
  const tokens = tokenize(ex.prompt)
  const heads = attentionHeadsFor(ex.prompt, tokens)
  const coref = heads.find((h) => h.label === 'coreference')
  expect(coref).toBeDefined()
  const itIdx = tokens.findIndex((t) => t.text.trim().toLowerCase() === 'it')
  const catIdx = tokens.findIndex((t) => t.text.trim().toLowerCase() === 'cat')
  expect(coref!.matrix[itIdx][catIdx]).toBeGreaterThanOrEqual(0.6)
  coref!.matrix.forEach((row, i) => {
    expect(row).toHaveLength(i + 1)
    expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-9)
  })
})

test('free prompts get the three procedural heads and no coreference', () => {
  const tokens = tokenize('Hello world out there')
  const heads = attentionHeadsFor('Hello world out there', tokens)
  expect(heads).toHaveLength(3)
  expect(heads.some((h) => h.label === 'coreference')).toBe(false)
})

test('coreference head is omitted when anchors are missing', () => {
  const ex = CURATED_EXAMPLES.find((e) => e.id === 'coreference')!
  const tokens = tokenize('no pronouns here at all')
  const heads = attentionHeadsFor(ex.prompt, tokens)  // prompt matches, tokens lack anchors
  expect(heads.some((h) => h.label === 'coreference')).toBe(false)
})

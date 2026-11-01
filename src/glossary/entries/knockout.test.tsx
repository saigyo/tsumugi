import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { getEntry } from '../registry'
import { GLOSSARY_IDS } from '../types'

afterEach(cleanup)

test('the knockout entry is registered and renders its caveats', () => {
  expect(GLOSSARY_IDS).toContain('knockout')
  const entry = getEntry('knockout')!
  expect(entry.stage).toBe('layers')
  expect(entry.related).toEqual(expect.arrayContaining(['votes', 'mlp', 'edit', 'sampling']))
  const { container } = render(<>{entry.body()}</>)
  const text = container.textContent ?? ''
  for (const word of ['decisive', 'preparatory', 'compensated', 'dampening', 'mean ablation', 'case study']) {
    expect(text).toContain(word)
  }
})

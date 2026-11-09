import { readFileSync } from 'node:fs'
import { expect, test } from 'vitest'

const css = readFileSync('src/index.css', 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')
const rootMatch = css.match(/:root\s*\{([\s\S]*?)\n\}/)
const rootBlock = rootMatch ? rootMatch[1] : ''
const rest = rootMatch ? css.replace(rootMatch[0], '') : css

const vars: Record<string, string> = {}
for (const m of rootBlock.matchAll(/(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;/g)) vars[m[1]] = m[2]

function luminance(hex: string): number {
  const channel = (i: number) => {
    const c = parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16) / 255
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2)
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

test('there is exactly one :root block', () => {
  expect(css.match(/:root\s*\{/g)).toHaveLength(1)
})

test('no colour literals outside :root', () => {
  expect(rest.match(/#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/g) ?? []).toEqual([])
})

test.each([
  ['--ink', '--washi'], ['--ink-soft', '--washi'], ['--ink-soft', '--card'],
  ['--ink-faint', '--washi'], ['--ink-faint', '--card'],
  ['--ai-mute', '--white'], ['--ai-mute', '--washi'], ['--ai-mute', '--card'],
  ['--on-ai', '--ai-deep'], ['--on-ai', '--shu'], ['--ai-deep', '--ai-wash'], ['--note-ink', '--card'],
])('%s on %s reaches 4.5:1', (fg, bg) => {
  expect(vars[fg], fg).toBeDefined()
  expect(vars[bg], bg).toBeDefined()
  expect(contrast(vars[fg], vars[bg])).toBeGreaterThanOrEqual(4.5)
})

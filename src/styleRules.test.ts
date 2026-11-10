import { readdirSync, readFileSync, statSync } from 'node:fs'
import { describe, expect, test } from 'vitest'

const css = readFileSync('src/index.css', 'utf8').replace(/\/\*[\s\S]*?\*\//g, '')
// The two variable blocks: light (:root) and dark (:root[data-theme="dark"]).
const rootMatch = css.match(/:root\s*\{([\s\S]*?)\n\}/)
const darkMatch = css.match(/:root\[data-theme="dark"\]\s*\{([\s\S]*?)\n\}/)
const rootBlock = rootMatch ? rootMatch[1] : ''
const rest = [rootMatch, darkMatch].reduce((text, m) => (m ? text.replace(m[0], '') : text), css)

const darkBlock = darkMatch ? darkMatch[1] : ''
const hexVars = (block: string) => {
  const out: Record<string, string> = {}
  for (const m of block.matchAll(/(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;/g)) out[m[1]] = m[2]
  return out
}
const themes: Record<string, Record<string, string>> = { light: hexVars(rootBlock), dark: hexVars(darkBlock) }
// Colour variables: those whose light value holds a colour literal or a shadow.
const colourNames = (block: string) =>
  [...block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)]
    .filter((m) => /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/.test(m[2]))
    .map((m) => m[1])
    .sort()

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

test('there are exactly two variable blocks: :root and :root[data-theme="dark"]', () => {
  expect(css.match(/:root\s*\{/g)).toHaveLength(1)
  expect(css.match(/:root\[data-theme="dark"\]\s*\{/g)).toHaveLength(1)
  expect(css.match(/:root[^{]*\{/g)).toHaveLength(2)
})

test('no colour literals outside the two variable blocks', () => {
  expect(rest.match(/#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/g) ?? []).toEqual([])
})

test('the dark block defines exactly the colour variables of the light block', () => {
  const light = colourNames(rootBlock)
  expect(light.length).toBeGreaterThan(40)
  expect(colourNames(darkBlock)).toEqual(light)
})

const pairs: [string, string][] = [
  ['--ink', '--washi'], ['--ink', '--card'], ['--ink-soft', '--washi'], ['--ink-soft', '--card'], ['--ink-soft', '--seg-ground'],
  ['--ink-faint', '--washi'], ['--ink-faint', '--card'],
  ['--ai-mute', '--white'], ['--ai-mute', '--washi'], ['--ai-mute', '--card'],
  ['--ai', '--washi'], ['--ai', '--card'],   // links and text in --ai
  ['--ai-deep', '--card'], ['--ai-deep', '--ai-wash'], ['--note-ink', '--card'],
  ['--ink', '--ai-wash-strong'],          // .head-score (active head chip)
  ['--ai-deep', '--ai-wash-strong'],      // active head chip, lit layer block
  ['--ink', '--dye-0'], ['--ink', '--dye-1'], ['--ink', '--dye-2'], ['--ink', '--dye-3'], ['--ink', '--dye-4'], ['--ink', '--dye-5'],  // token washes
  ['--shu-deep', '--shu-wash'],           // .token-end
  ['--ink-soft', '--ai-wash'],            // .run-chip-action on a hovered list row
  ['--ink-soft', '--shu-wash'],           // .run-chip-action on a compare A/B chip
  // text on the split fill roles
  ['--on-ai', '--fill'], ['--on-ai', '--fill-hover'], ['--on-shu', '--shu'], ['--on-fill-soft', '--fill-hover'],
]
describe.each(Object.keys(themes))('%s theme contrast', (theme) => {
  test.each(pairs)('%s on %s reaches 4.5:1', (fg, bg) => {
    const vars = themes[theme]
    expect(vars[fg], fg).toBeDefined()
    expect(vars[bg], bg).toBeDefined()
    expect(contrast(vars[fg], vars[bg])).toBeGreaterThanOrEqual(4.5)
  })
})

// Fill roles: text on indigo fills uses --on-ai over --fill / --fill-hover; text on vermillion uses --on-shu.
const blocks = [...rest.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ selector: m[1].trim().replace(/\s+/g, ' '), body: m[2] }))
const backgroundOf = (body: string) =>
  [...body.matchAll(/(?:^|[;\s])background(?:-color)?\s*:\s*([^;]+)/g)].map((m) => m[1].trim())

test('--on-ai text sits on --fill or --fill-hover, never on --ai-deep, --ai or --shu', () => {
  const bad = blocks
    .filter((b) => /color\s*:\s*var\(--on-ai\)/.test(b.body.replace(/background-color/g, '')))
    .filter((b) => backgroundOf(b.body).some((v) => /var\(--(ai-deep|ai|shu)\)/.test(v)))
    .map((b) => b.selector)
  expect(bad).toEqual([])
})

test('--on-shu text sits only on --shu', () => {
  const bad = blocks
    .filter((b) => /(?:^|[;\s])color\s*:\s*var\(--on-shu\)/.test(b.body))
    .filter((b) => backgroundOf(b.body).some((v) => v !== 'var(--shu)'))
    .map((b) => b.selector)
  expect(bad).toEqual([])
})

test('--on-ai is not used as a colour on vermillion', () => {
  const bad = blocks.filter((b) => /var\(--on-ai\)/.test(b.body) && backgroundOf(b.body).includes('var(--shu)'))
  expect(bad.map((b) => b.selector)).toEqual([])
})

// Type rules: everything outside :root and @font-face must use the shared variables.
const typeCss = rest.replace(/@font-face\s*\{[^}]*\}/g, '')
const declared = (prop: string) =>
  [...typeCss.matchAll(new RegExp(`(?:^|[;{\\s])${prop}\\s*:\\s*([^;}]+)`, 'g'))].map((m) => m[1].trim())

test('font sizes come from the scale', () => {
  const bad = declared('font-size')
    .filter((v) => !/^(var\(--t-(title|body|cap|label|tick|brand|prompt|mark|formula)\)|inherit)$/.test(v))
  expect(bad).toEqual([])
})

test('font families come from the four variables', () => {
  const bad = declared('font-family').filter((v) => !/^var\(--(sans|mono|mark|math)\)$/.test(v))
  expect(bad).toEqual([])
})

test('font weights are 400, 600 or 800, and font: is only ever inherit', () => {
  expect(declared('font-weight').filter((v) => !/^(400|600|800)$/.test(v))).toEqual([])
  expect(declared('font').filter((v) => v !== 'inherit')).toEqual([])
})

test('the type variables exist with the specified sizes', () => {
  const get = (name: string) => rootBlock.match(new RegExp(`${name}\\s*:\\s*([^;]+);`))?.[1].trim()
  expect(get('--t-title')).toBe('1.25rem')
  expect(get('--t-body')).toBe('.95rem')
  expect(get('--t-cap')).toBe('.82rem')
  expect(get('--t-label')).toBe('.72rem')
  expect(get('--t-tick')).toBe('.66rem')
  expect(get('--t-prompt')).toBe('1.3rem')
  for (const v of ['--t-brand', '--sans', '--mono', '--mark', '--math']) expect(get(v), v).toBeDefined()
})

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = `${dir}/${name}`
    if (statSync(path).isDirectory()) return sourceFiles(path)
    return /\.tsx?$/.test(name) && !/\.(test|spec)\.tsx?$/.test(name) ? [path] : []
  })
}

test('no colour literals in TypeScript source', () => {
  const colour = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/
  const offenders = sourceFiles('src')
    .filter((f) => colour.test(readFileSync(f, 'utf8').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '')))
  expect(offenders).toEqual([])
})

test('the light chart ramp end points are today\'s colours', () => {
  const get = (name: string) => rootBlock.match(new RegExp(`${name}\\s*:\\s*([^;]+);`))?.[1].trim()
  expect(get('--ramp-lo')).toBe('hsl(211 30% 94%)')
  expect(get('--ramp-hi')).toBe('hsl(211 55% 24%)')
  expect(get('--div-pos-lo')).toBe('hsl(211 45% 88%)')
  expect(get('--div-pos')).toBe('hsl(211 45% 38%)')
  expect(get('--div-neg-lo')).toBe('hsl(13 55% 90%)')
  expect(get('--div-neg')).toBe('hsl(13 55% 50%)')
})

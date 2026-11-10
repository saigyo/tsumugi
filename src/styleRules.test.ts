import { readdirSync, readFileSync, statSync } from 'node:fs'
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
  ['--ink', '--washi'], ['--ink-soft', '--washi'], ['--ink-soft', '--card'], ['--ink-soft', '--seg-ground'],
  ['--ink-faint', '--washi'], ['--ink-faint', '--card'],
  ['--ai-mute', '--white'], ['--ai-mute', '--washi'], ['--ai-mute', '--card'],
  ['--on-ai', '--ai-deep'], ['--on-ai', '--shu'], ['--ai-deep', '--ai-wash'], ['--note-ink', '--card'],
  // grounds the first set did not pair
  ['--ink-soft', '--seg-ground'],         // .mode-note
  ['--ink-soft', '--card'], ['--ink', '--ai-wash-strong'],   // .head-loc / .head-score (resting, active head chip)
  ['--ink', '--dye-2'],                   // .gl-stage-chip
  ['--shu-deep', '--shu-wash'],           // .token-end
  ['--ink-soft', '--ai-wash'],            // .run-chip-action on a hovered list row
  ['--ink-soft', '--shu-wash'],           // .run-chip-action on a compare A/B chip
  ['--on-ai', '--ai'],                    // hovered action on the active chip or row
])('%s on %s reaches 4.5:1', (fg, bg) => {
  expect(vars[fg], fg).toBeDefined()
  expect(vars[bg], bg).toBeDefined()
  expect(contrast(vars[fg], vars[bg])).toBeGreaterThanOrEqual(4.5)
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

// Colour values computed in TypeScript: only these five files, until the chart ramps move to variables.
const RAMP_FILES = [
  'src/viz/stripMath.ts', 'src/viz/mlpMath.ts', 'src/viz/AttentionHeatmap.tsx',
  'src/viz/AttentionGridExplorer.tsx', 'src/viz/EmbeddingGeometry.tsx',
]
function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = `${dir}/${name}`
    if (statSync(path).isDirectory()) return sourceFiles(path)
    return /\.tsx?$/.test(name) && !/\.(test|spec)\.tsx?$/.test(name) ? [path] : []
  })
}

test('no colour literals in TypeScript source outside the data ramps', () => {
  const colour = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/
  const offenders = sourceFiles('src')
    .filter((f) => !RAMP_FILES.includes(f))
    .filter((f) => colour.test(readFileSync(f, 'utf8').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '')))
  expect(offenders).toEqual([])
})

test('the allow-list holds only files that still need it', () => {
  // data-driven chart ramps; to be moved to variables with the dark theme
  const colour = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/
  for (const f of RAMP_FILES) expect(colour.test(readFileSync(f, 'utf8')), f).toBe(true)
})

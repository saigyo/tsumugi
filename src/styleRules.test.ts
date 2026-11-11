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
// Colour literals: hex and every colour function except color-mix (which mixes variables).
const colourLiteral = new RegExp(`#[0-9a-fA-F]{3,8}\\b|(?<![\\w-])(?:rgba?|hsla?|oklch|oklab|lab|lch|hwb|color)\\(`)
const NAMED = 'white|black|red|blue|green|gray|grey|orange|yellow|purple|pink'
const COLOUR_PROPS = '(?:color|background|background-color|border[\\w-]*|outline[\\w-]*|fill|stroke|box-shadow|text-decoration-color|caret-color|accent-color)'
// a named colour as a whole word in the value of a colour-taking property
const namedColourDecl = new RegExp(`(?:^|[;{\\s])${COLOUR_PROPS}\\s*:[^;}]*(?<![\\w-])(?:${NAMED})(?![\\w-])`, 'gm')
const colourNames = (block: string) =>
  [...block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)]
    .filter((m) => colourLiteral.test(m[2]))
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
  expect(rest.match(new RegExp(colourLiteral, 'g')) ?? []).toEqual([])
})

test('no named colours in colour-taking declarations outside the two variable blocks', () => {
  expect(rest.match(namedColourDecl) ?? []).toEqual([])
})

test('the colour guards flag literals and let variables and keywords through', () => {
  const flagged = (text: string) => colourLiteral.test(text) || new RegExp(namedColourDecl.source, 'm').test(text)
  for (const bad of ['color: white', 'fill: oklch(0.7 0.1 200)', 'background: #fff', 'border: 1px solid red', 'box-shadow: 0 0 2px rgb(0 0 0)',
    'stroke: color(srgb 1 0 0)', 'outline-color: hwb(200 10% 10%)', 'x: lab(50% 20 30)'])
    expect(flagged(bad), bad).toBe(true)
  for (const ok of ['color-mix(in hsl, var(--a) 50%, var(--b))', 'background: transparent', 'white-space: nowrap', 'color: currentColor',
    'border: none', 'color: inherit', 'background: var(--white)', 'fill: var(--ink)'])
    expect(flagged(ok), ok).toBe(false)
})

test('the example chip tag keeps its own card ground', () => {
  const tag = blocks.find((b) => b.selector === '.example-chip-tag')
  expect(tag && backgroundOf(tag.body)).toEqual(['var(--card)'])
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
  // .example-chip-tag is --ai on its own --card ground (guarded below), so the chip's hover wash is not behind its text
  ['--ai-deep', '--card'], ['--ai-deep', '--ai-wash'], ['--note-ink', '--card'],
  ['--ink', '--ai-wash-strong'],          // .head-score (active head chip)
  ['--ai-deep', '--ai-wash-strong'],      // active head chip, lit layer block
  ['--ink', '--dye-0'], ['--ink', '--dye-1'], ['--ink', '--dye-2'], ['--ink', '--dye-3'], ['--ink', '--dye-4'], ['--ink', '--dye-5'],  // token washes
  ['--shu-deep', '--shu-wash'],           // .token-end
  ['--ink-soft', '--ai-wash'],            // .run-chip-action on a hovered list row
  ['--ink', '--shu-wash'],                // is-chosen: the model's choice
  ['--shu-deep', '--ai-wash-strong'],     // the hovered action on a compare A/B chip (is-picked)
  ['--shu-deep', '--card'],                  // the pinned pin on a run chip
  // text on the split fill roles
  ['--on-ai', '--fill'], ['--on-ai', '--fill-hover'], ['--on-shu', '--shu'],
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

test('only the selected looks and the main button fill with --fill or --fill-hover', () => {
  const fills = blocks.filter((b) => backgroundOf(b.body).some((v) => /var\(--fill(-hover)?\)/.test(v)))
  const allowed = (selector: string) => selector.split(',').some((sel) => /^\.(is-view|btn-main)(?![\w-])/.test(sel.trim()))
    || selector === '.neuron-badge-strong'
  expect(fills.filter((b) => !allowed(b.selector)).map((b) => b.selector)).toEqual([])
  expect(fills.length).toBeGreaterThan(3)
})

test('the old one-off switch, tick and link selectors are gone', () => {
  for (const gone of ['.mlp-seg', '.cmp-mode-btn', '.layers-nav-link[aria-current="true"]', '.cmp-tick[data-selected="true"]'])
    expect(rest, gone).not.toContain(gone)
})

test('the three selected looks and the pill are defined as specified', () => {
  const body = (selector: string) => blocks.find((b) => b.selector === selector)?.body ?? ''
  expect(body('.is-picked, .is-picked:hover:not(:disabled)')).toMatch(/border-color: var\(--ai\);.*background: var\(--ai-wash-strong\);.*color: var\(--ai-deep\);.*font-weight: 600/)
  expect(body('.is-chosen, .is-chosen:hover:not(:disabled)')).toMatch(/border-color: var\(--shu\);.*background: var\(--shu-wash\)/)
  expect(body('.is-view:hover:not(:disabled)')).toMatch(/background: var\(--fill-hover\)/)
  expect(body('.pill:hover:not(:disabled)')).toMatch(/border-color: var\(--ai\)/)
})

test('the old chip and bar selectors are gone, and the vote colours are not used for bars', () => {
  const gone = [/\.token-chip(?![\w-])/, /\.hue-\d/, /\.hue-fill-\d/, /\.mlp-cand(?![\w-])/, /\.neuron-chip(?![\w-])/, /\.cmp-bar(?![\w-])/,
    /\.popover-bar(?![\w-])/, /\.votes-bar(?![\w-])/, /\.neuron-size-bar(?![\w-])/, /\.bar-rect(?![\w-])/]
  for (const re of gone) expect(rest, String(re)).not.toMatch(re)
  const usesVote = blocks.filter((b) => /--vote-pos/.test(b.body)).map((b) => b.selector)
  expect(usesVote.filter((sel) => !/^\.gl-vote-/.test(sel))).toEqual([])
})

test('the token chip and the bar are defined as specified', () => {
  const body = (selector: string) => blocks.find((b) => b.selector === selector)?.body ?? ''
  expect(body('.tok')).toMatch(/font-family: var\(--mono\);\s*font-size: var\(--t-cap\)/)
  expect(body('.tok')).toMatch(/border-radius: 999px; border: 1px solid var\(--line\); background: var\(--washi\)/)
  for (let i = 0; i < 6; i++) expect(body(`.tok-wash-${i}`)).toContain(`--wash: var(--dye-${i})`)
  expect(body('.tok.tok-away')).toMatch(/background: var\(--shu-wash\)/)
  expect(body('.bar')).toMatch(/height: 10px; border-radius: 2px; background: var\(--ai\)/)
  expect(body('.bar-chosen, .bar-neg')).toMatch(/background: var\(--shu\)/)
  expect(body('.bar-svg')).toMatch(/fill: var\(--ai\)/)
  expect(body('.bar-svg-chosen')).toMatch(/fill: var\(--shu\)/)
})

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
    .filter((v) => !/^(var\(--t-(title|body|cap|label|tick|brand|prompt|mark|formula|sub)\)|inherit)$/.test(v))
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
  expect(get('--t-sub')).toBe('1.05rem')
  for (const v of ['--t-brand', '--sans', '--mono', '--mark', '--math']) expect(get(v), v).toBeDefined()
})

test('corner radii come from the set', () => {
  const ok = /^(0|2px|4px|6px|8px|14px|999px|50%|2px 2px 0 0)$/
  expect(declared('border-radius').filter((v) => !ok.test(v))).toEqual([])
})
test('bold text is the semibold weight', () => {
  const rule = blocks.find((b) => b.selector === 'strong, b')
  expect(rule?.body).toMatch(/font-weight:\s*600/)
})

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = `${dir}/${name}`
    if (statSync(path).isDirectory()) return path === 'src/test' ? [] : sourceFiles(path)
    return /\.tsx?$/.test(name) && !/\.(test|spec)\.tsx?$/.test(name) ? [path] : []
  })
}

// Relative paths of the non-test TypeScript sources under src (outside src/test) whose text, comments left out, matches.
function filesWith(pattern: RegExp): string[] {
  return sourceFiles('src').filter((f) => pattern.test(readFileSync(f, 'utf8').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '')))
}

// "<button" opening tags (read to their closing ">", skipping braces and quotes) that carry no className attribute.
function buttonsWithoutClass(): string[] {
  const missing: string[] = []
  for (const file of sourceFiles('src')) {
    const text = readFileSync(file, 'utf8').replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, '')
    for (const start of text.matchAll(/<button\b/g)) {
      let depth = 0
      let quote = ''
      let end = start.index! + 7
      for (; end < text.length; end++) {
        const ch = text[end]
        if (quote) { if (ch === quote) quote = ''; continue }
        if (ch === '"' || ch === "'" || (depth > 0 && ch === '`')) quote = ch
        else if (ch === '{') depth++
        else if (ch === '}') depth--
        else if (ch === '>' && depth === 0 && text[end - 1] !== '=') break
      }
      if (!/\bclassName=/.test(text.slice(start.index!, end))) missing.push(`${file}: ${text.slice(start.index!, start.index! + 60).replace(/\s+/g, ' ')}`)
    }
  }
  return missing
}

test('the global button rule is a reset and has no hover rule', () => {
  expect(blocks.filter((b) => /^button(:[\w-]+(\([^)]*\))?)*$/.test(b.selector) && /:hover/.test(b.selector))).toEqual([])
  const base = blocks.find((b) => b.selector === 'button')
  expect(backgroundOf(base!.body)).toEqual(['none'])
})

test('every button in component code names its kind', () => {
  expect(buttonsWithoutClass()).toEqual([])
})

test('no colour literals in TypeScript source', () => {
  expect(filesWith(colourLiteral)).toEqual([])
})

test('no symbol or emoji icons in component code', () => {
  const banned = /[📌🔗⇩⇧⇄◐☀☾⏺⏸▶◀▾▸ⓘ]/u
  expect(filesWith(banned)).toEqual([])
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

test('the coral glow is used only by the main accent button and the active stage card', () => {
  const users = blocks.filter((b) => /var\(--shadow-accent\)/.test(b.body)).map((b) => b.selector)
  expect(users.sort()).toEqual(['.btn-accent', '.stage-card[data-active="true"], .stage-card[data-active="true"][data-clickable="true"]:hover'].sort())
})

test('the tooltip shadow variable is gone and the floating surface is one rule', () => {
  expect(rest).not.toContain('--shadow-tip')
  const body = blocks.find((b) => b.selector === '.pop')?.body ?? ''
  expect(body).toMatch(/background: var\(--card\)/)
  expect(body).toMatch(/color: var\(--ink\)/)
  expect(body).toMatch(/font-weight: 400/)
  expect(body).toMatch(/border: 1px solid var\(--line-strong\)/)
  expect(body).toMatch(/border-radius: 8px/)
  expect(body).toMatch(/box-shadow: var\(--shadow-pop\)/)
  expect(body).toMatch(/white-space: normal/)
  for (const sel of ['.gl-tip', '.step-box', '.token-popover', '.run-list']) {
    const own = blocks.find((b) => b.selector === sel)?.body ?? ''
    expect(own, sel).not.toMatch(/background|box-shadow|border/)
  }
})

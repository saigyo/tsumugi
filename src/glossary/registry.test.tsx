import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { sampleRun } from '../test/glossaryFixture'
import { EDIT_CALLOUTS, EMBEDDING_CALLOUTS, logitsCallouts, MLP_STEP_CALLOUTS, NEURON_CALLOUTS, SOFTMAX_CALLOUTS } from './callouts'
import { ENTRIES, getEntry, indexGroups, STAGE_ORDER } from './registry'
import { GLOSSARY_IDS } from './types'

afterEach(cleanup)

// raw sources of every entry file, for the inline <Term id section> check
const SOURCES = import.meta.glob('./entries/*.tsx', { query: '?raw', import: 'default', eager: true }) as Record<string, string>

const sectionsOf = (id: string): string[] => {
  const entry = getEntry(id)
  if (!entry) return []
  const { container } = render(<>{entry.body()}</>)
  const ids = [...container.querySelectorAll('[data-gl-section]')].map((el) => el.getAttribute('data-gl-section')!)
  cleanup()
  return ids
}

test('ids are unique and every entry has a known stage', () => {
  const ids = ENTRIES.map((e) => e.id)
  expect(new Set(ids).size).toBe(ids.length)
  for (const e of ENTRIES) expect(STAGE_ORDER).toContain(e.stage)
})

test('every summary is one sentence', () => {
  for (const e of ENTRIES) {
    expect(e.summary, e.id).toMatch(/\.$/)
    expect(e.summary.slice(0, -1), e.id).not.toMatch(/\. /)
  }
})

test('every body renders', () => {
  for (const e of ENTRIES) {
    const { container } = render(<>{e.body()}</>)
    expect(container.textContent?.length, e.id).toBeGreaterThan(80)
    cleanup()
  }
})

test('every inRun renders with a sample run and stays quiet without the data it needs', () => {
  const bare = sampleRun({ tokens: [], cycles: 0, stream: undefined, vocabSize: undefined, softmax: undefined, edits: undefined, votes: undefined })
  for (const e of ENTRIES) {
    if (!e.inRun) continue
    const { container } = render(<>{e.inRun(sampleRun())}</>)
    expect(container.textContent?.length, e.id).toBeGreaterThan(10)
    cleanup()
    const quiet = render(<>{e.inRun(bare)}</>)
    // entries whose block only needs mode/params may still speak; none may throw
    expect(quiet.container, e.id).toBeDefined()
    cleanup()
  }
})

test('inline <Term id section> links point at sections that exist', () => {
  const links = Object.values(SOURCES).flatMap((src) =>
    [...src.matchAll(/<Term id="([\w-]+)" section="([\w-]+)"/g)].map((m) => ({ id: m[1], section: m[2] })))
  for (const { id, section } of links) expect(sectionsOf(id), `${id} § ${section}`).toContain(section)
})

test('the index lists stages in pipeline order and leaves out how-to-read', () => {
  const groups = indexGroups()
  const order = groups.map((g) => STAGE_ORDER.indexOf(g.stage))
  expect(order).toEqual([...order].sort((a, b) => a - b))
  expect(groups.flatMap((g) => g.entries.map((e) => e.id))).not.toContain('how-to-read')
  expect(groups[0].label).toBe('Tokenizer')
})

test('the registry holds exactly one entry per id, in the declared order', () => {
  expect(ENTRIES.map((e) => e.id)).toEqual([...GLOSSARY_IDS])
})

test('every related id has an entry', () => {
  for (const e of ENTRIES) for (const r of e.related) expect(getEntry(r), `${e.id} → ${r}`).toBeDefined()
})

test('every callout chip points at a section that exists', () => {
  for (const c of [...EMBEDDING_CALLOUTS, ...logitsCallouts(3, '49 152'), ...SOFTMAX_CALLOUTS, ...EDIT_CALLOUTS, ...NEURON_CALLOUTS, ...MLP_STEP_CALLOUTS])
    expect(sectionsOf(c.id), `${c.label} → ${c.id} § ${c.section}`).toContain(c.section)
})

test('every callout chip hovers its own sentence, not the entry summary', () => {
  const sets = [EMBEDDING_CALLOUTS, logitsCallouts(3, '49 152'), SOFTMAX_CALLOUTS, EDIT_CALLOUTS, NEURON_CALLOUTS, MLP_STEP_CALLOUTS]
  const tips = sets.flat().map((c) => c.tip)
  for (const c of sets.flat()) {
    expect(c.tip.length, c.label).toBeGreaterThan(20)
    expect(c.tip, c.label).not.toBe(getEntry(c.id)!.summary)
  }
  expect(new Set(tips).size).toBe(tips.length)
})

test('How to read Tsumugi walks the screen in the order the app lays it out', () => {
  const { container } = render(<>{getEntry('how-to-read')!.body()}</>)
  const steps = [...container.querySelectorAll('li > strong')].map((el) => el.textContent)
  expect(steps).toEqual(['Prompt and mode.', 'Run shelf.', 'Generate.', 'Pipeline band.', 'Playback.', 'Detail panel.'])
})

test('entry formulas are typeset MathML and every one parses', () => {
  const withMath = ['embedding', 'residual-stream', 'edit', 'layer', 'attention', 'mlp', 'logit', 'softmax']
  for (const e of ENTRIES) {
    const { container } = render(<>{e.body()}</>)
    expect(container.querySelector('[data-formula-error]'), e.id).toBeNull()
    expect(container.querySelector('pre.gl-formula'), `${e.id} still has a monospace formula`).toBeNull()
    if (withMath.includes(e.id)) expect(container.querySelector('math'), e.id).not.toBeNull()
    cleanup()
  }
})

test('the MLP entry explains every symbol in its formula', () => {
  const { container } = render(<>{getEntry('mlp')!.body()}</>)
  const terms = [...container.querySelectorAll('dl.gl-symbols dt')].map((dt) => dt.querySelector('annotation')?.textContent ?? dt.textContent)
  for (const symbol of ['x', '\\operatorname{norm}', 'W_{\\text{gate}}', 'W_{\\text{up}}', '\\operatorname{SiLU}', '\\odot', 'W_{\\text{down}}'])
    expect(terms, symbol).toContain(symbol)
  // products are written with a visible operator, so "W n" cannot read as a subscript
  expect(container.querySelector('[data-testid="gl-mlp-steps"] annotation')?.textContent).toContain('W_{\\text{gate}} \\cdot n')
  expect(container.querySelector('[data-testid="gl-silu-plot"]')).not.toBeNull()
})

test('the logit formula writes its product with a visible operator', () => {
  const { container } = render(<>{getEntry('logit')!.body()}</>)
  expect(container.querySelector('annotation')?.textContent).toContain('\\operatorname{norm}(x_{\\text{last}}) \\cdot E^{\\top}')
})

test('the attention entry explains q, k, v and d and links to softmax', () => {
  const entry = getEntry('attention')!
  const { container } = render(<>{entry.body()}</>)
  const terms = [...container.querySelectorAll('dl.gl-symbols dt')].map((dt) => dt.querySelector('annotation')?.textContent ?? dt.textContent)
  for (const symbol of ['q_i', 'k_j', 'v_j', 'd', 'a_{ij}', 'o_i', 'W_O'])
    expect(terms, symbol).toContain(symbol)
  expect(container.querySelector('[data-term="softmax"]')).not.toBeNull()
  expect(entry.related).toContain('softmax')
})

test('wide formulas are broken into rows so they fit the drawer', () => {
  const tex = (id: string) => {
    const { container } = render(<>{getEntry(id)!.body()}</>)
    const t = [...container.querySelectorAll('div.formula annotation')].map((a) => a.textContent ?? '').join('\n')
    cleanup()
    return t
  }
  // attention: q, k and v on rows of their own
  expect(tex('attention')).toMatch(/q_i &= W_Q \\cdot n_i \\\\[\s\S]*k_j &= W_K[\s\S]*v_j &= W_V/)
  // logit: the shape line is its own row, not an indented continuation
  expect(tex('logit')).not.toContain('\\phantom')
})

test('the MLP gate scales the values rather than just letting them through', () => {
  const { container } = render(<>{getEntry('mlp')!.body()}</>)
  expect(container.textContent).toContain('scaled by the gate')
})

test('the votes entry explains the frozen norm and how it differs from the logit lens', () => {
  const { container } = render(<>{getEntry('votes')!.body()}</>)
  expect(container.textContent).toMatch(/logit lens/i)
  expect(container.textContent).toMatch(/root mean square/i)
  expect(container.querySelector('[data-formula-error]')).toBeNull()
})

test('the MLP entry no longer says the app hides its internals', () => {
  const { container } = render(<>{getEntry('mlp')!.body()}</>)
  expect(container.textContent).not.toMatch(/does not show the MLP’s internals/)
  expect(container.textContent).toMatch(/What each layer voted for/)
})

test('In this run lines show whitespace tokens with their glyphs', () => {
  const run = sampleRun({ edits: { token: { id: 198, text: '\n' }, layer: 3, ratio: 0.5 }, votes: { candidate: { id: 198, text: '\n' }, layer: 29, value: 7.6 } })
  for (const id of ['edit', 'votes']) {
    const { container } = render(<>{getEntry(id)!.inRun!(run)}</>)
    expect(container.textContent, id).toContain('“↵”')
    cleanup()
  }
})

test('the votes entry builds the idea in steps, with a figure of two edits and their shares', () => {
  const { container } = render(<>{getEntry('votes')!.body()}</>)
  expect(container.querySelectorAll('ol.gl-steps > li')).toHaveLength(5)
  expect(container.querySelector('[data-testid="gl-votes-figure"]')).not.toBeNull()
  expect(container.textContent).toMatch(/\(a \+ b\) · E = a · E \+ b · E/)
})

test('the MLP neuron entry explains the split step by step and links from MLP and votes', () => {
  const { container } = render(<>{getEntry('mlp-neuron')!.body()}</>)
  expect(container.querySelectorAll('ol.gl-steps > li')).toHaveLength(4)
  expect(sectionsOf('mlp-neuron')).toEqual(['rarely-names', 'negative', 'early-layers'])
  expect(getEntry('mlp')!.related).toContain('mlp-neuron')
  expect(getEntry('votes')!.related).toContain('mlp-neuron')
})

test('the chat template entry has its five sections and sits in the Tokenizer group', () => {
  expect(sectionsOf('chat-template')).toEqual(['what', 'system-prompt', 'special-tokens', 'continue-text', 'sink'])
  expect(getEntry('chat-template')?.stage).toBe('tokenize')
})

test('the generation settings entry has a section per setting and sits with Sampling', () => {
  expect(sectionsOf('generation-settings')).toEqual(['temperature', 'top-k', 'max-new-tokens'])
  const entry = getEntry('generation-settings')!
  expect(entry.stage).toBe('sample')
  expect(entry.related).toEqual(expect.arrayContaining(['sampling', 'softmax']))
  expect(getEntry('sampling')!.related).toContain('generation-settings')
})

test('generation settings in this run: the run’s own values, how many tokens it wrote and why it stopped', () => {
  const inRun = getEntry('generation-settings')!.inRun!
  const { container } = render(<>{inRun(sampleRun({ generated: 5, ended: 'eos' }))}</>)
  expect(container.textContent).toBe('This run: T = 0.8, top-k 10, up to 8 new tokens. It wrote 5 and stopped because the model picked its end token.')
  cleanup()
  render(<>{inRun(sampleRun({ generated: 8, ended: 'max-tokens' }))}</>)
  expect(document.body.textContent).toContain('It wrote 8 and stopped at the limit.')
  cleanup()
  render(<>{inRun(sampleRun({ params: { temperature: 0, topK: 10, maxNewTokens: 8 }, generated: 3, ended: undefined }))}</>)
  expect(document.body.textContent).toBe('This run: T = 0 (greedy), top-k 10, up to 8 new tokens. 3 so far.')
  cleanup()
  render(<>{inRun(sampleRun({ generated: 2, ended: 'aborted' }))}</>)
  expect(document.body.textContent).toContain('It wrote 2 and was stopped by hand.')
})

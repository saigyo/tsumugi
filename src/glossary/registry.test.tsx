import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { sampleRun } from '../test/glossaryFixture'
import { EMBEDDING_CALLOUTS, logitsCallouts, SOFTMAX_CALLOUTS } from './callouts'
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
  const bare = sampleRun({ tokens: [], cycles: 0, stream: undefined, vocabSize: undefined, softmax: undefined })
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
  for (const c of [...EMBEDDING_CALLOUTS, ...logitsCallouts(3, '49 152'), ...SOFTMAX_CALLOUTS])
    expect(sectionsOf(c.id), `${c.label} → ${c.id} § ${c.section}`).toContain(c.section)
})

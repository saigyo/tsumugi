import { errors, expect, test, type Locator, type Page } from '@playwright/test'
import { installProbe, type Control, type ControlState } from './polish/probe'
import { setTheme, THEMES, walk, type Theme } from './polish/walk'

test.setTimeout(240_000)
test.use({ viewport: { width: 1380, height: 940 } })

const STATES = [
  'first-visit', 'sim-run-end', 'sim-embeddings', 'sim-layers', 'sim-logits', 'sim-softmax', 'sim-sampler',
  'settings-tip', 'term-tip', 'glossary-index', 'glossary-softmax', 'glossary-mlp', 'runs-8', 'runs-list',
  'compare-armed', 'compare-side', 'compare-combined', 'real-layers',
]

test.beforeEach(async ({ page }) => {
  await installProbe(page)
})

/** Walks the app once per theme and hands each state to `visit`. */
async function inEachTheme(page: Page, visit: (theme: Theme, state: string) => Promise<void>) {
  for (const theme of THEMES) {
    await setTheme(page, theme)
    await walk(page, (state) => visit(theme, state))
  }
}

/** A theme that checks fewer controls than this was not really surveyed (a full walk reaches about 90). */
const MIN_CONTROLS = 20

/** Controls that may legitimately never be reachable by the pointer. Empty on purpose: add an entry only with a reason. */
const NEVER_REACHABLE: string[] = []

const stateOf = (page: Page, index: number) => page.evaluate((i) => window.__polish.state(i), index)

/**
 * Runs `act` on one representative of every distinct control not yet reached in this theme.
 * A control the pointer cannot reach in time (Playwright timeout) is noted and may be reached in a later state;
 * any other error fails the test. `finish()` lists the controls that were never reached.
 */
function reachTracker(page: Page) {
  const reached = new Set<string>()
  const skipped = new Map<string, string>()
  return {
    async eachNew(state: string, act: (control: Control, locator: Locator) => Promise<void>) {
      const controls = await page.evaluate(() => window.__polish.controls())
      for (const control of controls) {
        if (reached.has(control.key)) continue
        const locator = page.locator(`[data-polish="${control.index}"]`)
        try {
          await locator.scrollIntoViewIfNeeded({ timeout: 1000 })
          await act(control, locator)
        } catch (error) {
          if (!(error instanceof errors.TimeoutError)) throw error
          if (!skipped.has(control.key)) skipped.set(control.key, state)
          continue
        }
        reached.add(control.key)
      }
    },
    finish(): string[] {
      return [...skipped]
        .filter(([key]) => !reached.has(key) && !NEVER_REACHABLE.includes(key))
        .map(([key, state]) => `never reachable: ${key} (first seen in ${state})`)
    },
  }
}

type HoverRecord = {
  theme: Theme
  state: string
  control: Control
  rest: ControlState | null
  hovered: ControlState | null
  moved: string[]
}
type HoverSurvey = { records: HoverRecord[]; palettes: Record<Theme, Record<string, string>>; unreachable: string[] }

let hoverSurvey: Promise<HoverSurvey> | undefined

/** Walks and hovers every distinct control once per theme. The three hover tests share this record; the memo is per worker. */
function surveyHovers(page: Page): Promise<HoverSurvey> {
  hoverSurvey ??= (async () => {
    const survey: HoverSurvey = { records: [], palettes: {} as HoverSurvey['palettes'], unreachable: [] }
    for (const theme of THEMES) {
      const tracker = reachTracker(page)
      await setTheme(page, theme)
      await walk(page, async (state) => {
        survey.palettes[theme] = await page.evaluate(() => window.__polish.palette())
        await tracker.eachNew(state, async (control, locator) => {
          await page.mouse.move(2, 2)
          await page.waitForTimeout(60)
          const rest = await stateOf(page, control.index)
          await page.evaluate(() => window.__polish.snapshot())
          await locator.hover({ timeout: 1000 })
          await page.waitForTimeout(250) // the 0.2s hover transitions
          const hovered = await stateOf(page, control.index)
          const moved = await page.evaluate((i) => window.__polish.movedVertically(i), control.index)
          survey.records.push({ theme, state, control, rest, hovered, moved })
        })
      })
      const surveyed = survey.records.filter((r) => r.theme === theme).length
      if (surveyed < MIN_CONTROLS) throw new Error(`${theme}: the hover survey reached only ${surveyed} controls, expected at least ${MIN_CONTROLS}`)
      survey.unreachable.push(...tracker.finish().map((m) => `${theme}: ${m}`))
    }
    return survey
  })().catch((error) => {
    hoverSurvey = undefined
    throw error
  })
  return hoverSurvey
}

const label = (r: HoverRecord) => `${r.theme}/${r.state}: ${r.control.key} "${r.control.text}"`

test('the walk reaches every state', async ({ page }) => {
  for (const theme of THEMES) {
    const seen: string[] = []
    await setTheme(page, theme)
    await walk(page, async (state) => { seen.push(state) })
    expect(seen, theme).toEqual(STATES)
  }
})

test.fixme('hovered controls stay readable', async ({ page }) => {
  const survey = await surveyHovers(page)
  const problems = [...survey.unreachable]
  for (const r of survey.records) {
    if (!r.hovered) problems.push(`${label(r)} vanished when hovered`)
    else if (!r.hovered.disabled && (r.hovered.contrast ?? 0) < 4.5)
      problems.push(`${label(r)} contrast ${r.hovered.contrast} (${r.hovered.text} on ${r.hovered.background})`)
  }
  expect(problems).toEqual([])
})

test.fixme('hover keeps fills and chart colours', async ({ page }) => {
  const survey = await surveyHovers(page)
  const problems = [...survey.unreachable]
  for (const r of survey.records) {
    if (!r.rest || !r.hovered) {
      problems.push(`${label(r)} vanished when hovered`)
      continue
    }
    const palette = survey.palettes[r.theme]
    const neutral = [palette.card, palette.washi, palette.white, palette['seg-ground'], 'transparent']
    if (neutral.includes(r.rest.background)) continue
    const allowed = r.rest.background === palette.fill ? [r.rest.background, palette['fill-hover']] : [r.rest.background]
    if (!allowed.includes(r.hovered.background))
      problems.push(`${label(r)} background ${r.rest.background} became ${r.hovered.background}, expected ${allowed.join(' or ')}`)
  }
  expect(problems).toEqual([])
})

test.fixme('hovering moves nothing vertically', async ({ page }) => {
  const survey = await surveyHovers(page)
  const problems = [...survey.unreachable]
  for (const r of survey.records) {
    if (!r.hovered) problems.push(`${label(r)} vanished when hovered`)
    else if (r.moved.length) problems.push(`${label(r)} moved ${r.moved.slice(0, 3).join('; ')}`)
  }
  expect(problems).toEqual([])
})

test.fixme('every control has an accessible name', async ({ page }) => {
  const problems = new Set<string>()
  await inEachTheme(page, async (theme, state) => {
    const { checked, unnamed } = await page.evaluate(() => window.__polish.unnamedControls())
    if (checked === 0) problems.add(`${theme}/${state}: no controls were checked`)
    for (const control of unnamed) problems.add(`${theme}/${state}: ${control} has no accessible name`)
  })
  expect([...problems]).toEqual([])
})

test.fixme('every control shows a focus ring', async ({ page }) => {
  const problems: string[] = []
  for (const theme of THEMES) {
    const tracker = reachTracker(page)
    let checkedCount = 0
    await setTheme(page, theme)
    await walk(page, async (state) => {
      await tracker.eachNew(state, async (control, locator) => {
        await page.mouse.move(2, 2)
        await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur()) // an autofocused field is not "resting"
        await page.waitForTimeout(60)
        const before = await page.evaluate((i) => window.__polish.ring(i), control.index)
        await page.keyboard.press('Shift') // so that script focus counts as keyboard focus
        await locator.focus({ timeout: 1000 })
        await page.waitForTimeout(250) // focus transitions
        const after = await page.evaluate((i) => window.__polish.ring(i), control.index)
        if (!before || !after) problems.push(`${theme}/${state}: ${control.key} "${control.text}" vanished when focused`)
        else if (!after.visible || after.signature === before.signature)
          problems.push(`${theme}/${state}: ${control.key} "${control.text}" shows no focus ring (${after.signature})`)
        await locator.blur()
        checkedCount++
      })
    })
    if (checkedCount < MIN_CONTROLS) problems.push(`${theme}: only ${checkedCount} controls were checked, expected at least ${MIN_CONTROLS}`)
    problems.push(...tracker.finish().map((m) => `${theme}: ${m}`))
  }
  expect(problems).toEqual([])
})

test.fixme('numbers follow one format', async ({ page }) => {
  const problems: string[] = []
  await inEachTheme(page, async (theme, state) => {
    for (const issue of await page.evaluate(() => window.__polish.textIssues())) problems.push(`${theme}/${state}: ${issue}`)
  })
  expect(problems).toEqual([])
})

test.fixme('chart labels are not cut', async ({ page }) => {
  const problems: string[] = []
  await inEachTheme(page, async (theme, state) => {
    for (const issue of await page.evaluate(() => window.__polish.clippedLabels())) problems.push(`${theme}/${state}: ${issue}`)
  })
  expect(problems).toEqual([])
})

test.fixme('tooltip text stays inside its box', async ({ page }) => {
  const problems: string[] = []
  await inEachTheme(page, async (theme, state) => {
    if (state !== 'settings-tip' && state !== 'term-tip') return
    const tips = await page.getByTestId('gl-tip').count()
    if (!tips) problems.push(`${theme}/${state}: no tooltip is showing`)
    for (const issue of await page.evaluate(() => window.__polish.overflowingTips())) problems.push(`${theme}/${state}: ${issue}`)
  })
  expect(problems).toEqual([])
})

test.fixme('the page never scrolls sideways at 400px and 1380px', async ({ page }) => {
  const problems: string[] = []
  for (const width of [400, 1380]) {
    await page.setViewportSize({ width, height: 940 })
    await inEachTheme(page, async (theme, state) => {
      const over = await page.evaluate(() => window.__polish.sidewaysScroll())
      if (over !== 0) problems.push(`${theme}/${state} at ${width}px: scrolls ${over}px sideways`)
    })
  }
  expect(problems).toEqual([])
})

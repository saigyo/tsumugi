import { expect, test, type Locator, type Page } from '@playwright/test'
import { installProbe, type Control } from './polish/probe'
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

/** Runs `act` on one representative of every distinct control not yet seen in this theme. Controls that cannot be reached by the pointer are skipped. */
async function eachNewControl(
  page: Page,
  seen: Set<string>,
  act: (control: Control, locator: Locator) => Promise<void>,
): Promise<number> {
  let reached = 0
  const controls = await page.evaluate(() => window.__polish.controls())
  for (const control of controls) {
    if (seen.has(control.key)) continue
    const locator = page.locator(`[data-polish="${control.index}"]`)
    try {
      await locator.scrollIntoViewIfNeeded({ timeout: 1000 })
      await act(control, locator)
    } catch {
      continue
    }
    seen.add(control.key)
    reached++
  }
  return reached
}

const stateOf = (page: Page, index: number) => page.evaluate((i) => window.__polish.state(i), index)

/** Moves the pointer away, records the resting state, hovers, and lets the 0.2s transitions end. */
async function restAndHover(page: Page, control: Control, locator: Locator) {
  await page.mouse.move(2, 2)
  await page.waitForTimeout(60)
  const rest = await stateOf(page, control.index)
  await locator.hover({ timeout: 1000 })
  await page.waitForTimeout(250)
  const hovered = await stateOf(page, control.index)
  return { rest, hovered }
}

test('the walk reaches every state', async ({ page }) => {
  for (const theme of THEMES) {
    const seen: string[] = []
    await setTheme(page, theme)
    await walk(page, async (state) => { seen.push(state) })
    expect(seen, theme).toEqual(STATES)
  }
})

test.fixme('hovered controls stay readable', async ({ page }) => {
  const problems: string[] = []
  for (const theme of THEMES) {
    const seen = new Set<string>()
    await setTheme(page, theme)
    await walk(page, async (state) => {
      await eachNewControl(page, seen, async (control, locator) => {
        const { hovered } = await restAndHover(page, control, locator)
        if (hovered && !hovered.disabled && (hovered.contrast ?? 21) < 4.5)
          problems.push(`${theme}/${state}: ${control.key} "${control.text}" contrast ${hovered.contrast} (${hovered.text} on ${hovered.background})`)
      })
    })
  }
  expect(problems).toEqual([])
})

test.fixme('hover keeps fills and chart colours', async ({ page }) => {
  const problems: string[] = []
  for (const theme of THEMES) {
    const seen = new Set<string>()
    await setTheme(page, theme)
    let palette: Record<string, string> = {}
    await walk(page, async (state) => {
      palette = await page.evaluate(() => window.__polish.palette())
      const neutral = new Set([palette.card, palette.washi, palette.white, palette['seg-ground'], 'transparent'])
      await eachNewControl(page, seen, async (control, locator) => {
        const { rest, hovered } = await restAndHover(page, control, locator)
        if (!rest || !hovered || neutral.has(rest.background)) return
        const allowed = rest.background === palette.fill ? [rest.background, palette['fill-hover']] : [rest.background]
        if (!allowed.includes(hovered.background))
          problems.push(`${theme}/${state}: ${control.key} "${control.text}" background ${rest.background} became ${hovered.background}, expected ${allowed.join(' or ')}`)
      })
    })
  }
  expect(problems).toEqual([])
})

test.fixme('hovering moves nothing vertically', async ({ page }) => {
  const problems: string[] = []
  for (const theme of THEMES) {
    const seen = new Set<string>()
    await setTheme(page, theme)
    await walk(page, async (state) => {
      await eachNewControl(page, seen, async (control, locator) => {
        await page.mouse.move(2, 2)
        await page.waitForTimeout(60)
        await page.evaluate(() => window.__polish.snapshot())
        await locator.hover({ timeout: 1000 })
        await page.waitForTimeout(250)
        const moved = await page.evaluate((i) => window.__polish.movedVertically(i), control.index)
        if (moved.length) problems.push(`${theme}/${state}: hovering ${control.key} "${control.text}" moved ${moved.slice(0, 3).join('; ')}`)
      })
    })
  }
  expect(problems).toEqual([])
})

test.fixme('every control has an accessible name', async ({ page }) => {
  const problems: string[] = []
  for (const theme of THEMES) {
    const seen = new Set<string>()
    await setTheme(page, theme)
    await walk(page, async (state) => {
      const controls = await page.evaluate(() => window.__polish.controls())
      for (const control of controls) {
        if (seen.has(control.key)) continue
        seen.add(control.key)
        const s = await stateOf(page, control.index)
        if (s && !s.hasName) problems.push(`${theme}/${state}: ${control.key} has no accessible name`)
      }
    })
  }
  expect(problems).toEqual([])
})

test.fixme('every control shows a focus ring', async ({ page }) => {
  const problems: string[] = []
  for (const theme of THEMES) {
    const seen = new Set<string>()
    await setTheme(page, theme)
    await walk(page, async (state) => {
      await eachNewControl(page, seen, async (control, locator) => {
        await page.keyboard.press('Shift') // so that script focus counts as keyboard focus
        await locator.focus({ timeout: 1000 })
        const s = await stateOf(page, control.index)
        if (s && !s.focusRing) problems.push(`${theme}/${state}: ${control.key} "${control.text}" shows no focus ring`)
        await locator.blur()
      })
    })
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

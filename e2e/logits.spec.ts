import { expect, test, type Page } from '@playwright/test'
import { openFixtureRun } from './fixtureRun'

// The score list is a table: chip, product strip, bar, number. Engines size a chip column differently (WebKit once let a
// long chip run into the strip), so this runs in all three.

// Long and short tokens, as a real vocabulary has them; the second is longer than a chip may get.
const WORDS = ['␣native', '␣internationalization', '␣a']

async function scoreRows(page: Page, width: number) {
  await page.setViewportSize({ width, height: 900 })
  await openFixtureRun(page)
  await page.locator('[data-testid="stage-card"][data-stage="logits"]').click()
  await expect(page.getByTestId('readout-candidate').first()).toBeVisible()
  return page.evaluate((words) => {
    const rows = [...document.querySelectorAll('[data-testid="readout-candidate"]')]
    rows.forEach((row, i) => { row.querySelector('.chip-text')!.textContent = words[i % words.length] })
    const equation = document.querySelector('.readout-equation')!
    return {
      overflow: equation.scrollWidth - equation.clientWidth,
      rows: rows.map((row) => {
        const [chip, strip, bar, value] = [...row.children].map((c) => c.getBoundingClientRect())
        return { chipRight: chip.right, stripLeft: strip.left, barLeft: bar.left, valueRight: value.right }
      }),
    }
  }, WORDS)
}

// 1280: the equation in one row. 980: one row, just above the width where it stacks. 700: stacked.
for (const width of [1280, 980, 700]) {
  test(`the score list keeps its columns at ${width}px`, async ({ page }) => {
    const { rows, overflow } = await scoreRows(page, width)
    expect(rows.length).toBeGreaterThan(2)
    for (const key of ['stripLeft', 'barLeft', 'valueRight'] as const) {
      const at = rows.map((r) => Math.round(r[key]))
      expect(new Set(at).size, `${key}: ${at.join(' ')}`).toBe(1)
    }
    for (const r of rows) expect(r.chipRight).toBeLessThanOrEqual(r.stripLeft - 4)
    expect(overflow).toBeLessThanOrEqual(0)
  })
}

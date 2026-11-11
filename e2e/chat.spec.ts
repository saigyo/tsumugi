import { expect, test } from '@playwright/test'
import { importChatRun } from './fixtureRun'

test('an imported chat run: folded strip, end of turn, and an attention matrix that fits its card', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/')
  await expect(page.getByTestId('sim-banner')).toBeVisible()
  await expect(page.getByTestId('mode-badge')).toHaveText('Simulated')
  await page.getByTestId('prompt-input').fill('seed run')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })

  await importChatRun(page)

  await expect(page.getByTestId('token-stream').getByTestId('system-fold')).toHaveText('system prompt · 20 tokens')
  await expect(page.getByTestId('generated-token').last()).toHaveClass(/token-end/)
  await expect(page.getByTestId('generated-token').filter({ hasText: 'Paris' })).toHaveCount(1)

  await page.locator('[data-testid="stage-card"][data-stage="layers"]').click()
  const heatmap = page.getByTestId('attention-heatmap')
  // the cell size follows the card's measured width: wait until it has settled
  await expect.poll(async () => {
    const card = await heatmap.boundingBox()
    const svg = await heatmap.locator('svg').boundingBox()
    return card && svg ? card.width - svg.width : -1
  }).toBeGreaterThanOrEqual(0)
  await expect(page.getByTestId('attn-show-system')).not.toBeChecked()
  await expect(page.getByTestId('edit-fold')).toHaveText('system prompt · 20 tokens')
})

// The band is at its limit with this system's fonts, and other systems have wider ones (Linux: about 6% wider),
// so the check runs once as it is and once with a font that is wider still.
for (const font of [null, 'Verdana']) {
  test(`the pipeline band fits its column in a chat run at 1024px${font ? ', also with a wider font' : ''}`, async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 900 })
    await page.goto('/')
    await page.getByTestId('prompt-input').fill('seed run')
    await page.getByTestId('btn-generate').click()
    await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })
    await importChatRun(page)
    await expect(page.getByTestId('token-stream').getByTestId('system-fold')).toBeVisible()
    if (font) await page.addStyleTag({ content: `:root { --sans: ${font}, sans-serif !important; }` })
    const band = page.getByTestId('pipeline-band')
    await expect(band).toBeVisible()
    // every step of the run has its own summaries, so check the band at each of them
    const back = page.getByTestId('btn-step-back')
    const wide: string[] = []
    const paddings: number[] = []
    for (let step = 0; step < 300; step++) {
      // the band refits itself after a change; give it the frames it needs before reading
      await expect.poll(() => band.evaluate((el) => el.scrollWidth - el.clientWidth), { timeout: 2000 }).toBeLessThanOrEqual(0)
        .catch(() => {})
      const { scroll, client } = await band.evaluate((el) => ({ scroll: el.scrollWidth, client: el.clientWidth }))
      if (scroll > client) wide.push(`step ${step}: ${scroll} > ${client}`)
      paddings.push(await band.locator('.stage-card').first().evaluate((el) => parseFloat(getComputedStyle(el).paddingLeft)))
      if (await back.isDisabled()) break
      await back.click()
    }
    expect(wide).toEqual([])
    // within one run the cards never get their padding back: they do not change shape from step to step
    const grew = paddings.flatMap((p, i) => (i > 0 && p > paddings[i - 1] + 0.01 ? [`step ${i}: ${paddings[i - 1]} to ${p}`] : []))
    expect(grew).toEqual([])
    if (font) expect(new Set(paddings).size).toBeLessThanOrEqual(3)
  })
}

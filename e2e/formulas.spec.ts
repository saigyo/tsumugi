import { expect, test, type Page } from '@playwright/test'

async function openSoftmax(page: Page) {
  await page.goto('/')
  await page.getByTestId('example-chip').first().click()
  await page.getByTestId('run-chip').first().waitFor({ timeout: 60_000 })
  await page.getByTestId('btn-live').click()
  await page.getByTestId('stage-card').nth(4).click()
  const pause = page.getByTestId('btn-pause')
  if (await pause.count()) await pause.first().click()
  await page.getByTestId('softmax-formula').waitFor()
  await page.evaluate(() => document.fonts.ready)
  await page.waitForTimeout(450)
}

test('the fraction line of the Softmax formula is drawn', async ({ page }) => {
  await openSoftmax(page)
  const frac = page.getByTestId('softmax-formula').locator('mfrac').first()
  await frac.scrollIntoViewIfNeeded()
  const box = (await frac.boundingBox())!
  const num = (await frac.locator('> *').nth(0).boundingBox())!
  const den = (await frac.locator('> *').nth(1).boundingBox())!
  const png = (await frac.screenshot()).toString('base64')
  const best = await page.evaluate(async ({ png, top, bottom, scaleBox }) => {
    const img = new Image()
    img.src = `data:image/png;base64,${png}`
    await img.decode()
    const k = img.width / scaleBox
    const c = document.createElement('canvas')
    c.width = img.width; c.height = img.height
    const g = c.getContext('2d')!
    g.drawImage(img, 0, 0)
    const data = g.getImageData(0, 0, c.width, c.height).data
    let best = 0
    for (let y = Math.floor(top * k); y <= Math.ceil(bottom * k) && y < c.height; y++) {
      let dark = 0
      for (let x = 0; x < c.width; x++) {
        const i = (y * c.width + x) * 4
        if (data[i + 3] > 0 && data[i] + data[i + 1] + data[i + 2] < 3 * 128) dark++
      }
      best = Math.max(best, dark / c.width)
    }
    return best
  }, { png, top: num.y + num.height - box.y, bottom: den.y - box.y, scaleBox: box.width })
  expect(best).toBeGreaterThanOrEqual(0.6)
})

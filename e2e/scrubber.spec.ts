import { expect, test } from '@playwright/test'

// Engines differ in how a range input follows the mouse (with a list of tick positions, Firefox and Chromium
// snap the thumb to them), so this runs in all three.
test('dragging the scrubber reaches the steps between two tokens, and every stage', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/')
  await page.getByTestId('maxtok-input').fill('40')
  await page.getByTestId('prompt-input').fill('The cat sat on the')
  await page.getByTestId('btn-generate').click()
  await page.getByTestId('btn-pause').click()
  const scrubber = page.getByTestId('scrubber')
  const box = (await scrubber.boundingBox())!
  const y = box.y + box.height / 2
  await page.mouse.move(box.x + 60, y)
  await page.mouse.down()
  const values = new Set<number>()
  const stages = new Set<string>()
  for (let dx = 60; dx <= 260; dx++) {
    await page.mouse.move(box.x + dx, y)
    values.add(Number(await scrubber.inputValue()))
    stages.add(await page.locator('.stage-card[data-active="true"]').getAttribute('data-stage') ?? '')
  }
  await page.mouse.up()
  // 200 pixels of a track that holds more steps than pixels: most steps under the mouse are reached, not only
  // the one at the end of each token
  const span = Math.max(...values) - Math.min(...values) + 1
  expect(span).toBeGreaterThan(150)
  expect(values.size).toBeGreaterThan(span * 0.6)
  expect([...stages].sort()).toEqual(['embeddings', 'layers', 'logits', 'sampler', 'softmax'])
})

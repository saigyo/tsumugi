import { expect, test, type Page } from '@playwright/test'

// Engines differ in how a range input follows the mouse and the keyboard (with a list of tick positions, Firefox
// and Chromium snap the thumb to them), so this runs in all three.

async function pausedRun(page: Page) {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/')
  await page.getByTestId('maxtok-input').fill('40')
  await page.getByTestId('prompt-input').fill('The cat sat on the')
  await page.getByTestId('btn-generate').click()
  await page.getByTestId('btn-pause').click()
  return page.getByTestId('scrubber')
}

// "Step 12 of 723"
const stepOf = async (scrubber: ReturnType<Page['getByTestId']>) =>
  Number((await scrubber.getAttribute('aria-valuetext'))?.match(/Step (\d+) of/)?.[1])

test('dragging the scrubber reaches every stage, each with room in proportion to its playback time', async ({ page }) => {
  const scrubber = await pausedRun(page)
  const box = (await scrubber.boundingBox())!
  const y = box.y + box.height / 2
  await page.mouse.move(box.x + 60, y)
  await page.mouse.down()
  const steps = new Set<number>()
  const stages = new Map<string, number>()
  let positions = 0
  for (let dx = 60; dx <= 360; dx++) {
    await page.mouse.move(box.x + dx, y)
    steps.add(await stepOf(scrubber))
    const stage = await page.locator('.stage-card[data-active="true"]').getAttribute('data-stage') ?? ''
    stages.set(stage, (stages.get(stage) ?? 0) + 1)
    positions++
  }
  await page.mouse.up()
  expect([...stages.keys()].sort()).toEqual(['embeddings', 'layers', 'logits', 'sampler', 'softmax'])
  // not only the step at the end of each token (a snapping thumb), but the steps between
  expect(steps.size).toBeGreaterThan(100)
  // Embeddings, Logits and Softmax are one step in eighteen each, but get about a ninth of the track each,
  // because playback stays on them longer than on a layer step
  for (const stage of ['embeddings', 'logits', 'softmax']) {
    expect(stages.get(stage)! / positions, stage).toBeGreaterThan(0.09)
  }
})

test('on the scrubber an arrow key moves exactly one step', async ({ page }) => {
  const scrubber = await pausedRun(page)
  // a few steps in, so that the start is a step that is shown
  for (let i = 0; i < 5; i++) await page.getByTestId('btn-step-fwd').click()
  await scrubber.focus()
  const from = await stepOf(scrubber)
  expect(from).toBeGreaterThan(3)
  await page.keyboard.press('ArrowRight')
  await expect.poll(() => stepOf(scrubber)).toBe(from + 1)
  await page.keyboard.press('ArrowRight')
  await page.keyboard.press('ArrowRight')
  await expect.poll(() => stepOf(scrubber)).toBe(from + 3)
  await page.keyboard.press('ArrowLeft')
  await expect.poll(() => stepOf(scrubber)).toBe(from + 2)
})

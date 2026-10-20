import { expect, test, type Page } from '@playwright/test'
import { buildFixtureTrace } from '../src/test/fixtures'
import { neuronFixtureFile } from '../src/test/neuronFixture'

async function openFixtureRun(page: Page) {
  await page.route(/huggingface\.co\/.*\/neurons\//, async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').pop()!
    const body = neuronFixtureFile(name)
    await route.fulfill(body
      ? { status: 200, body: Buffer.from(body), headers: { 'access-control-allow-origin': '*' },
          contentType: name.endsWith('.json') ? 'application/json' : 'application/octet-stream' }
      : { status: 404, headers: { 'access-control-allow-origin': '*' } })
  })
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('seed run')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true, prompt: 'steps fixture' })
  const file = JSON.stringify({ formatVersion: 1, events, meta: { prompt: 'steps fixture',
    params: { temperature: 0.8, topK: 10, maxNewTokens: 2 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(), reason: 'max-tokens' } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  await page.getByTestId('run-chip').filter({ hasText: 'steps fixture' }).getByTestId('run-chip-main').click()
  await page.getByTestId('stage-card').nth(2).click()
}

test('the section bar sticks, jumps and steps the shared layer; the charts fill the card', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await openFixtureRun(page)
  const nav = page.getByTestId('layers-nav')
  await expect(nav).toBeVisible()

  await page.getByTestId('nav-link').filter({ hasText: 'Step by step' }).click()
  await expect(page.getByTestId('steps-title')).toBeInViewport()
  await expect(page.getByTestId('nav-link').filter({ hasText: 'Step by step' })).toHaveAttribute('aria-current', 'true')
  // Both bars are stuck: the playback bar at the top, the section bar directly below it.
  // Poll, because the jump scrolls smoothly and the bars settle only when it ends.
  await expect.poll(async () => {
    const c = (await page.locator('.controls').boundingBox())!
    const n = (await nav.boundingBox())!
    return Math.abs(n.y - (c.y + c.height)) <= 1 && Math.round(c.y) === 0
  }).toBe(true)

  const layerOf = async (id: string) => Number((await page.getByTestId(id).textContent())!.match(/layer (\d+)/)![1])
  const before = await layerOf('steps-title')
  const expected = before > 0 ? before - 1 : before + 1
  await page.getByTestId(before > 0 ? 'nav-prev' : 'nav-next').click()
  await expect.poll(() => layerOf('steps-title')).toBe(expected)
  await expect.poll(() => layerOf('neuron-title')).toBe(expected)

  await expect(page.getByTestId('strip-n')).toBeVisible()
  await expect(page.getByTestId('steps-scatter')).toBeVisible()
  await expect(page.getByTestId('steps-silu')).toBeVisible()
  await expect.poll(async () => (await page.getByTestId('steps-scatter').boundingBox())!.width).toBeGreaterThanOrEqual(700)
  await expect.poll(async () => {
    const scatter = (await page.getByTestId('steps-scatter').boundingBox())!
    const silu = (await page.getByTestId('steps-silu').boundingBox())!
    return Math.abs(scatter.x - silu.x) < 1 && Math.abs(scatter.width - silu.width) < 1
  }).toBe(true)
})

test('at phone width the page does not scroll sideways', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await openFixtureRun(page)
  await page.getByTestId('nav-link').filter({ hasText: 'Step by step' }).click()
  await expect(page.getByTestId('steps-scatter')).toBeVisible()
  await expect(page.locator('.controls')).toBeVisible()
  await expect(page.getByTestId('layers-nav')).toBeVisible()
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  expect(overflow).toBeLessThanOrEqual(0)
})

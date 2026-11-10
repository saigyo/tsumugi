import { expect, test } from '@playwright/test'
import { openFixtureRun } from './fixtureRun'

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

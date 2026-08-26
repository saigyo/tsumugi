import { expect, test } from '@playwright/test'

test('simulated run produces tokens and steps through stages', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('The cat sat on the')
  await page.getByTestId('btn-generate').click()

  // playback auto-starts; a generated token eventually lands in the stream
  await expect(page.getByTestId('generated-token').first()).toBeVisible({ timeout: 15000 })

  // pause and manually step: exactly one stage card is active
  await page.getByTestId('btn-pause').click()
  await page.getByTestId('btn-step-fwd').click()
  const activeCards = page.locator('[data-testid="stage-card"][data-active="true"]')
  await expect(activeCards).toHaveCount(1)
})

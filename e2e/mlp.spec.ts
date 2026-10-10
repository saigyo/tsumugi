import { expect, test } from '@playwright/test'

test('simulated run: the Layers detail shows both MLP cards with the real-model-only note', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('The cat sat on the')
  await page.getByTestId('btn-generate').click()
  await page.getByTestId('run-chip').first().waitFor({ timeout: 30000 })   // the run has sealed
  await page.getByTestId('stage-card').nth(2).click()                      // Layers
  await expect(page.getByTestId('edit-size-note')).toHaveText('Real model only: simulated runs have no MLP to measure.')
  await expect(page.getByTestId('votes-note')).toHaveText('Real model only: simulated runs have no MLP to measure.')
  await page.getByTestId('votes-info').click()
  await expect(page.getByTestId('gl-heading')).toContainText('Per-layer votes')
})

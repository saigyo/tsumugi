import { expect, test } from '@playwright/test'

test('first visit: hint opens How to read; ? opens the index; a card chip opens its entry with run numbers', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByTestId('glossary-hint')).toBeVisible()
  await page.getByTestId('btn-hint-open').click()
  await expect(page.getByTestId('gl-heading')).toContainText('How to read Tsumugi')
  await expect(page.getByTestId('glossary-hint')).toHaveCount(0)
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('glossary-drawer')).toHaveCount(0)

  await page.getByTestId('btn-help').click()
  await expect(page.getByTestId('gl-heading')).toHaveText('Glossary')
  await page.getByTestId('gl-close').click()

  // a simulated run, then the Softmax card's temperature chip
  await page.getByTestId('prompt-input').fill('The cat sat on the')
  await page.getByTestId('btn-generate').click()
  await page.getByTestId('run-chip').first().waitFor({ timeout: 30000 })   // the run has sealed
  await page.locator('[data-testid="stage-card"][data-stage="softmax"]').click()
  await page.getByTestId('embed-callout').filter({ hasText: 'temperature' }).click()
  await expect(page.getByTestId('gl-heading')).toContainText('Softmax')
  await expect(page.getByTestId('gl-in-run')).toContainText('T =')
})

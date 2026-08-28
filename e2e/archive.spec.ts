import { expect, test } from '@playwright/test'

test('runs archive to the shelf, compare opens and exits', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('one two three')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })

  await page.getByTestId('prompt-input').fill('red green blue')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(2, { timeout: 15000 })

  await page.getByTestId('btn-compare-arm').click()
  await page.getByTestId('run-chip-main').first().click()
  await expect(page.getByTestId('compare-view')).toBeVisible()
  await expect(page.getByTestId('cmp-badge')).toContainText('different prompts')

  await page.getByTestId('btn-compare-exit').click()
  await expect(page.getByTestId('compare-view')).not.toBeVisible()
  await expect(page.getByTestId('stage-card').first()).toBeVisible()
})

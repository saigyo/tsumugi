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
  await page.getByTestId('run-chip-main').last().click()
  await expect(page.getByTestId('compare-view')).toBeVisible()
  await expect(page.getByTestId('cmp-badge')).toContainText('different prompts')

  await page.getByTestId('btn-compare-exit').click()
  await expect(page.getByTestId('compare-view')).not.toBeVisible()
  await expect(page.getByTestId('stage-card').first()).toBeVisible()
})

test('chip actions are reachable by keyboard', async ({ page }) => {
  await page.goto('/')
  for (const prompt of ['one two three', 'red green blue']) {
    const n = await page.getByTestId('run-chip').count()
    await page.getByTestId('prompt-input').fill(prompt)
    await page.getByTestId('btn-generate').click()
    await expect(page.getByTestId('run-chip')).toHaveCount(n + 1, { timeout: 15000 })
  }
  // the newest run is active; the other chip is the second one in the row
  const other = page.locator('[data-testid="run-chip"][data-active="false"]')
  await expect(other).toHaveCount(1)
  const remove = other.getByTestId('btn-chip-remove')
  await expect(remove).toBeHidden()
  // the active chip's remove button is the last stop before the other chip's main button
  await page.locator('[data-testid="run-chip"][data-active="true"]').getByTestId('btn-chip-remove').focus()
  await page.keyboard.press('Tab')
  await expect(other.getByTestId('run-chip-main')).toBeFocused()
  await expect(remove).toBeVisible()
  // Tab through pin, (share), export until the remove button holds focus
  for (let i = 0; i < 5 && !(await remove.evaluate((el) => el === document.activeElement)); i++) await page.keyboard.press('Tab')
  await expect(remove).toBeFocused()
  await expect(remove).toBeVisible()
})

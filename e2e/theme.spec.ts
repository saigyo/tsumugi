import { expect, test } from '@playwright/test'

test.use({ colorScheme: 'dark' })

test('theme follows the system, the switch overrides it and the choice survives a reload', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' })
  await page.evaluate(() => localStorage.clear())
  await page.reload({ waitUntil: 'domcontentloaded' })
  // set by the inline script, before the app module has run
  expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark')

  const toggle = page.getByTestId('theme-toggle')
  const html = page.locator('html')
  await toggle.click()
  await expect(html).toHaveAttribute('data-theme', 'light')
  await page.reload({ waitUntil: 'domcontentloaded' })
  expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe('light')

  await toggle.click() // dark
  await toggle.click() // auto
  await expect(toggle).toHaveAttribute('aria-label', 'Theme: auto (follows system)')
  await expect(html).toHaveAttribute('data-theme', 'dark')
})

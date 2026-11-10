import { expect, test } from '@playwright/test'

test.use({ colorScheme: 'dark' })

test('theme follows the system, the switch overrides it and the choice survives a reload', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' })
  await page.evaluate(() => localStorage.clear())
  await page.reload({ waitUntil: 'domcontentloaded' })
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

test('the inline script in index.html sets the theme before the app module runs', async ({ page }) => {
  // the app module never loads, so only the inline script can have set the attribute
  await page.route(/\/src\/main\.tsx/, (route) => route.abort())
  await page.goto('/', { waitUntil: 'domcontentloaded' })
  expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark')
  expect(await page.getByTestId('theme-toggle').count()).toBe(0) // the app did not start
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
  expect(bg).not.toBe('rgb(255, 255, 255)')
})

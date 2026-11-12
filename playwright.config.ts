import { defineConfig, devices } from '@playwright/test'

// The design check (polish.spec.ts) walks the app in both themes and takes minutes, so it is a project of its
// own: `npm run e2e` (and CI, which the deploy waits for) runs the other Chromium tests, `npm run e2e:design`
// (and the "Design check" workflow) runs the walk in Chromium.
// Firefox and WebKit run only the checks that depend on rendering differences: `npm run e2e:browsers`.
const design = /polish\.spec\.ts/
const crossEngine = [design, /formulas\.spec\.ts/, /scrubber\.spec\.ts/, /logits\.spec\.ts/]

export default defineConfig({
  testDir: 'e2e',
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: 'http://localhost:5173', actionTimeout: 15_000 },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] }, testIgnore: design },
    { name: 'design', use: { ...devices['Desktop Chrome'] }, testMatch: design },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] }, testMatch: crossEngine },
    { name: 'webkit', use: { ...devices['Desktop Safari'] }, testMatch: crossEngine },
  ],
  webServer: { command: 'npm run dev', url: 'http://localhost:5173', reuseExistingServer: true },
})

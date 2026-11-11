import { defineConfig, devices } from '@playwright/test'

// Firefox and WebKit run only the checks that depend on rendering differences.
// `npm run e2e` (and CI) runs Chromium; `npm run e2e:browsers` runs the other two.
const crossEngine = [/polish\.spec\.ts/, /formulas\.spec\.ts/]

export default defineConfig({
  testDir: 'e2e',
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: 'http://localhost:5173', actionTimeout: 15_000 },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] }, testMatch: crossEngine },
    { name: 'webkit', use: { ...devices['Desktop Safari'] }, testMatch: crossEngine },
  ],
  webServer: { command: 'npm run dev', url: 'http://localhost:5173', reuseExistingServer: true },
})

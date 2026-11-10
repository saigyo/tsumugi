import { expect, type Page } from '@playwright/test'
import { buildFixtureTrace } from '../src/test/fixtures'
import { neuronFixtureFile } from '../src/test/neuronFixture'

export async function openFixtureRun(page: Page) {
  await page.route(/huggingface\.co\/.*\/neurons\//, async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').pop()!
    const body = neuronFixtureFile(name)
    await route.fulfill(body
      ? { status: 200, body: Buffer.from(body), headers: { 'access-control-allow-origin': '*' },
          contentType: name.endsWith('.json') ? 'application/json' : 'application/octet-stream' }
      : { status: 404, headers: { 'access-control-allow-origin': '*' } })
  })
  await page.goto('/')
  await page.getByTestId('prompt-input').fill('seed run')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true, prompt: 'steps fixture' })
  const file = JSON.stringify({ formatVersion: 1, events, meta: { prompt: 'steps fixture',
    params: { temperature: 0.8, topK: 10, maxNewTokens: 2 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(), reason: 'max-tokens' } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  await page.getByTestId('run-chip').filter({ hasText: 'steps fixture' }).getByTestId('run-chip-main').click()
  await page.locator('[data-testid="stage-card"][data-stage="layers"]').click()
}

import { expect, test } from '@playwright/test'
import { buildFixtureTrace } from '../src/test/fixtures'
import { neuronFixtureFile } from '../src/test/neuronFixture'

test('an imported real run opens Inside one MLP with a total equal to the votes card cell', async ({ page }) => {
  await page.route(/huggingface\.co\/.*\/neurons\//, async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').pop()!
    const body = neuronFixtureFile(name)
    await route.fulfill(body
      ? { status: 200, body: Buffer.from(body), headers: { 'access-control-allow-origin': '*' },
          contentType: name.endsWith('.json') ? 'application/json' : 'application/octet-stream' }
      : { status: 404, headers: { 'access-control-allow-origin': '*' } })
  })
  await page.goto('/')
  // The run shelf (and its import button) only appears once the archive holds a run,
  // so archive one simulated run first.
  await page.getByTestId('prompt-input').fill('seed run')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true, prompt: 'neuron fixture' })
  const file = JSON.stringify({ formatVersion: 1, events, meta: { prompt: 'neuron fixture',
    params: { temperature: 0.8, topK: 10, maxNewTokens: 2 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(), reason: 'max-tokens' } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  await page.getByTestId('run-chip').filter({ hasText: 'neuron fixture' }).getByTestId('run-chip-main').click()
  await page.locator('[data-testid="stage-card"][data-stage="layers"]').click()

  const card = page.getByTestId('neuron-card')
  await expect(card.getByTestId('neuron-row')).toHaveCount(8)
  const layer = Number((await card.getByTestId('neuron-title').textContent())!.match(/layer (\d+)/)![1])
  const cell = await page.locator(`[data-testid="vote-cell"][data-row="0"][data-layer="${layer}"]`).getAttribute('data-value')
  const total = (await card.getByTestId('neuron-total').textContent())!.replace('−', '-')
  expect(Number(total.match(/[+-]\d+\.\d{2}/)![0])).toBeCloseTo(Number(cell), 2)

  await card.getByTestId('neuron-mode-toggle').nth(1).click()
  await expect(card.getByTestId('neuron-table')).toContainText('edit size')
  const stepper = layer < 2 ? 'neuron-next' : 'neuron-prev'
  await card.getByTestId(stepper).click()
  await expect(card.getByTestId('neuron-title')).toContainText(`layer ${layer < 2 ? layer + 1 : layer - 1}`)
})

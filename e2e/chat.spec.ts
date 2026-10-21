import { expect, test } from '@playwright/test'
import { buildFixtureTrace } from '../src/test/fixtures'

test('an imported chat run: folded strip, end of turn, and an attention matrix that fits its card', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/')
  await expect(page.getByTestId('sim-banner')).toBeVisible()
  await expect(page.getByTestId('mode-badge')).toHaveText('Simulated')
  await page.getByTestId('prompt-input').fill('seed run')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })

  const events = buildFixtureTrace({ chat: true, mode: 'real', mlp: true })
  const file = JSON.stringify({ formatVersion: 1, events, meta: { prompt: 'What is the capital of France?',
    params: { temperature: 0, topK: 10, maxNewTokens: 8 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(),
    reason: 'eos', format: { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' } } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  const chip = page.getByTestId('run-chip').filter({ hasText: 'Chat · What is the' })
  await chip.getByTestId('run-chip-main').click()

  await expect(page.getByTestId('system-fold')).toHaveText('system prompt · 20 tokens ▸')
  await expect(page.getByTestId('generated-token').last()).toHaveClass(/token-end/)
  await expect(page.getByTestId('generated-token').filter({ hasText: 'Paris' })).toHaveCount(1)

  await page.getByTestId('stage-card').nth(2).click()
  const heatmap = page.getByTestId('attention-heatmap')
  const card = (await heatmap.boundingBox())!
  const svg = (await heatmap.locator('svg').boundingBox())!
  expect(svg.width).toBeLessThanOrEqual(card.width)
  await expect(page.getByTestId('attn-show-system')).not.toBeChecked()
  await expect(page.getByTestId('edit-fold')).toHaveText('system prompt · 20 tokens ▸')
})

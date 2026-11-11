import { expect, type Page } from '@playwright/test'
import { buildFixtureTrace, fixtureSweep } from '../src/test/fixtures'
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
  // sweepable, with a knockout sweep for each cycle: the Knockouts card has bars once a model that can measure is announced (see announceKnockoutModel)
  const events = buildFixtureTrace({ mode: 'real', mlp: true, neurons: true, sweepable: true, prompt: 'steps fixture' })
  const sweeps = [0, 1].map((cycle) => fixtureSweep(events, cycle))
  const file = JSON.stringify({ formatVersion: 1, events, sweeps, meta: { prompt: 'steps fixture',
    params: { temperature: 0.8, topK: 10, maxNewTokens: 2 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(), reason: 'max-tokens' } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  await page.getByTestId('run-chip').filter({ hasText: 'steps fixture' }).getByTestId('run-chip-main').click()
  await page.locator('[data-testid="stage-card"][data-stage="layers"]').click()
}

/** Imports the chat example run ("What is the capital of France?") and opens it. The page must already be loaded. */
export async function importChatRun(page: Page) {
  const events = buildFixtureTrace({ chat: true, mode: 'real', mlp: true })
  const file = JSON.stringify({ formatVersion: 1, events, meta: { prompt: 'What is the capital of France?',
    params: { temperature: 0, topK: 10, maxNewTokens: 8 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(),
    reason: 'eos', format: { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' } } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  await page.getByTestId('run-chip').filter({ hasText: 'What is the' }).first().getByTestId('run-chip-main').click()
}

/**
 * Tells the app that a real model able to measure knockouts is loaded, without downloading one: the Knockouts card shows
 * its stored sweep only then. The page must already be loaded (a reload forgets it). The dev server serves the app's own
 * module, so this sets the state the app reads. The string form keeps the import out of Playwright's transpiler.
 */
export async function announceKnockoutModel(page: Page) {
  await page.evaluate(`import('/src/viz/knockoutRequest.ts').then((m) => m.useKnockoutRequest.setState({ model: { knockout: true, revision: 'v7' } }))`)
}

const GEOMETRY_VOCAB = 49152
const GEOMETRY_K = 4
const GEOMETRY_PCA = 8

/**
 * Serves a small stand-in for the vocabulary geometry asset (the Hub is not reachable from the check), written for the
 * fixture model id, so the Embeddings card of an imported run draws its neighbour list and its similarity grid.
 * Vectors are smooth in the token id, so neighbouring ids are similar and distant ones are not.
 */
export async function serveGeometry(page: Page) {
  const q = new Int8Array(GEOMETRY_VOCAB * GEOMETRY_PCA)
  for (let id = 0; id < GEOMETRY_VOCAB; id++)
    for (let d = 0; d < GEOMETRY_PCA; d++) q[id * GEOMETRY_PCA + d] = Math.round(100 * Math.sin(id * 0.013 * (d + 1) + d))
  const neighbors = Buffer.alloc(GEOMETRY_VOCAB * GEOMETRY_K * 3)
  for (let id = 0; id < GEOMETRY_VOCAB; id++)
    for (let i = 0; i < GEOMETRY_K; i++) {
      neighbors.writeUInt16LE((id + i + 1) % GEOMETRY_VOCAB, (id * GEOMETRY_K + i) * 2)
      neighbors.writeUInt8(240 - 10 * i, GEOMETRY_VOCAB * GEOMETRY_K * 2 + id * GEOMETRY_K + i)
    }
  const tokens = Array.from({ length: GEOMETRY_VOCAB }, (_, id) => `t${id}`)
  const manifest = { modelId: 'fixture', vocabSize: GEOMETRY_VOCAB, dims: 576, k: GEOMETRY_K, pcaDims: GEOMETRY_PCA, scale: 1 / 127,
    explainedVariance: 1, sourceSha256: '0'.repeat(64), files: {} }
  const cors = { 'access-control-allow-origin': '*' }
  await page.route(/huggingface\.co\/.*\/geometry\//, async (route) => {
    const name = new URL(route.request().url()).pathname.split('/').pop()
    if (name === 'manifest.json') return route.fulfill({ status: 200, json: manifest, headers: cors })
    if (name === 'tokens.json') return route.fulfill({ status: 200, json: tokens, headers: cors })
    if (name === 'neighbors.bin') return route.fulfill({ status: 200, body: neighbors, headers: cors, contentType: 'application/octet-stream' })
    if (name === 'vectors.bin') return route.fulfill({ status: 200, body: Buffer.from(q.buffer), headers: cors, contentType: 'application/octet-stream' })
    return route.fulfill({ status: 404, headers: cors })
  })
}

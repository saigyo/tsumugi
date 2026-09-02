// Regenerates docs/screenshots/*.png against the current build.
//
//   npm run screenshots
//
// Spawns its own dev server on port 5199 and drives a headed Chromium with a
// persistent profile (.screenshots-profile/, gitignored) so the ~120 MB real
// model stays cached across invocations — only the first run downloads it.
// Real-mode shots need WebGPU, which is why the browser runs headed.
//
// Notes for future edits:
// - Clicking a curated example chip AUTO-GENERATES (v1 behavior). Never click
//   btn-generate right after an example chip — that starts a second run.
// - The compare shot wants a same-prompt fork; sampling may coincide, so the
//   script regenerates until the fork tick appears (up to 4 attempts).
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { chromium } from '@playwright/test'

const PORT = 5199
const URL = `http://localhost:${PORT}/`
const DIR = new globalThis.URL('../docs/screenshots', import.meta.url).pathname
const PROFILE = new globalThis.URL('../.screenshots-profile', import.meta.url).pathname

const waitForServer = async () => {
  for (let i = 0; i < 60; i++) {
    try { if ((await fetch(URL)).ok) return } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`dev server did not come up on :${PORT}`)
}

mkdirSync(DIR, { recursive: true })
const server = spawn('npm', ['run', 'dev', '--', '--port', String(PORT)], { stdio: 'ignore' })
const context = await chromium.launchPersistentContext(PROFILE, {
  headless: false,
  viewport: { width: 1380, height: 940 },
})
// Scenes can be selected: `npm run screenshots -- pipeline attention compare embeddings`
// (default: all). Skipping a scene preserves its existing images — useful when
// a previous capture caught a specimen worth keeping (e.g. a great fork).
const only = process.argv.slice(2)
const want = (scene) => only.length === 0 || only.includes(scene)

try {
  await waitForServer()
  const page = context.pages()[0] ?? await context.newPage()
  await page.goto(URL)
  const $ = (id) => page.getByTestId(id)

  // clean shelf (the profile persists IndexedDB, so old runs survive reloads);
  // wait for hydration first — records that land after a clear would be re-merged
  await page.waitForTimeout(1500)
  while (await $('btn-chip-remove').count()) await $('btn-chip-remove').first().click()

  // ---- pipeline.png: simulated run at run end -------------------------------
  if (want('pipeline')) {
    await $('example-chip').first().click()                  // auto-generates
    await $('run-chip').first().waitFor({ timeout: 15000 })
    await $('btn-live').click()
    await page.screenshot({ path: `${DIR}/pipeline.png` })
    console.log('pipeline.png')
  }

  // real mode serves both remaining scenes
  if (want('attention') || want('compare') || want('embeddings')) {
    await $('mode-toggle').check()
    await $('model-status-slot').getByText(/attn/).waitFor({ timeout: 300000 })  // first run downloads the model
  }

  // ---- real-attention.png + head-explorer.png: real mode --------------------
  if (want('attention')) {
    const before = await $('run-chip').count()
    await $('example-chip').first().click()
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })
    await $('btn-live').click()
    await $('stage-card').nth(2).click()                     // Layers
    await $('detail-layers').scrollIntoViewIfNeeded()
    await page.screenshot({ path: `${DIR}/real-attention.png` })
    console.log('real-attention.png')
    await $('btn-explore-heads').click()
    await $('grid-explorer').scrollIntoViewIfNeeded()
    await page.waitForTimeout(500)                           // let the canvases paint
    await page.screenshot({ path: `${DIR}/head-explorer.png` })
    console.log('head-explorer.png')
  }

  // ---- compare.png + compare-attention.png: same-prompt fork ----------------
  if (want('compare')) {
  await $('prompt-input').fill('The cat sat on the mat because it was tired')
  await $('temp-input').fill('0.9')
  {
    const before = await $('run-chip').count()               // baseline T=0.9 run,
    await $('btn-generate').click()                          // so the pair shares T
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })
  }
  for (let attempt = 0; attempt < 4; attempt++) {
    const before = await $('run-chip').count()
    await $('btn-generate').click()                          // same prompt again
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })  // wait for the run to SEAL
    await $('btn-compare-arm').click()
    await $('run-chip-main').nth(before - 1).click()         // B = previous run
    await $('compare-view').waitFor({ timeout: 5000 })
    if (await page.locator('[data-testid="cmp-tick"][data-fork="true"]').count()) break
    console.log('no fork this time — regenerating')
    await $('btn-compare-exit').click()
  }
  const fork = page.locator('[data-testid="cmp-tick"][data-fork="true"]')
  await ((await fork.count()) ? fork.first() : $('cmp-tick').nth(2)).click()
  await $('cmp-dist-side').first().waitFor({ timeout: 5000 })
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: `${DIR}/compare.png` })
  console.log('compare.png')
  await $('cmp-attn-side').first().scrollIntoViewIfNeeded()
  await page.screenshot({ path: `${DIR}/compare-attention.png` })
  console.log('compare-attention.png')
  }

  // ---- embeddings.png: real mode, Embeddings stage — lookup + neighbours + matrix
  if (want('embeddings')) {
    await $('prompt-input').fill('The cat sat on the mat because the cat was tired')
    await $('maxtok-input').fill('8')                               // short run: the card stays one screen tall
    const before = await $('run-chip').count()
    await $('btn-generate').click()
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })   // wait for the run to SEAL
    await $('btn-live').click()
    await $('stage-card').nth(1).click()                            // Embeddings
    await $('embed-neighbors').waitFor({ timeout: 60000 })          // geometry asset fetched from the Hub
    // the card (chips + strips + neighbours + matrix) is taller than the default
    // viewport; crop to the playback controls plus the card, like the other shots
    await page.setViewportSize({ width: 1380, height: 1500 })
    const controls = page.locator('.controls')
    await controls.evaluate((el) => el.scrollIntoView({ block: 'start' }))
    await page.waitForTimeout(300)
    const top = await controls.boundingBox()
    const card = await $('detail-embeddings').boundingBox()
    const pad = 12
    const topPad = 4   // tight: the pipeline band's loop arrow sits just above the controls
    await page.screenshot({ path: `${DIR}/embeddings.png`, clip: {
      x: card.x - pad, y: top.y - topPad, width: card.width + 2 * pad, height: card.y + card.height - top.y + topPad + pad,
    } })
    await page.setViewportSize({ width: 1380, height: 940 })
    await $('maxtok-input').fill('20')                              // restore the default for later scenes
    console.log('embeddings.png')
  }
} finally {
  await context.close()
  server.kill()
}

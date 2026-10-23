// Regenerates docs/screenshots/*.png against the current build.
//
//   npm run screenshots
//
// Spawns its own dev server on port 5199 and drives a headed Chromium with a
// persistent profile (.screenshots-profile/, gitignored) so the ~250 MB real
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
// detached: the server gets its own process group, so the cleanup below can kill
// npm AND the vite child it spawns — killing npm alone left vite on :5199
const server = spawn('npm', ['run', 'dev', '--', '--port', String(PORT)], { stdio: 'ignore', detached: true })
const stopServer = () => { try { process.kill(-server.pid, 'SIGTERM') } catch { /* already gone */ } }
process.on('exit', stopServer)
const context = await chromium.launchPersistentContext(PROFILE, {
  headless: false,
  viewport: { width: 1380, height: 940 },
})
// Scenes can be selected: `npm run screenshots -- glossary pipeline attention compare embeddings logits mlp neurons`
// (mlp also writes mlp-steps.png)
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

  // ---- glossary.png: simulated run, Softmax card with the drawer open -------
  if (want('glossary')) {
    await $('example-chip').first().click()                  // auto-generates
    await $('run-chip').first().waitFor({ timeout: 15000 })
    await $('btn-live').click()
    await $('stage-card').nth(4).click()                     // Softmax
    await $('embed-callout').filter({ hasText: 'why exponentiate' }).click()
    await $('gl-in-run').waitFor({ timeout: 5000 })
    await page.waitForTimeout(800)                           // let the section scroll and the highlight settle
    await page.screenshot({ path: `${DIR}/glossary.png` })
    await $('gl-close').click()
    console.log('glossary.png')
  }

  // real mode serves the remaining scenes
  if (want('pipeline') || want('attention') || want('compare') || want('embeddings') || want('logits') || want('mlp') || want('neurons')) {
    await $('mode-real').click()
    await $('mode-badge').filter({ hasText: 'attn' }).waitFor({ timeout: 300000 })  // first run downloads the model
  }

  // ---- pipeline.png: real model, the chat example at run end ----------------
  if (want('pipeline')) {
    await $('temp-input').fill('0')                          // greedy: the answer ends its turn with <|im_end|>
    const before = await $('run-chip').count()
    await $('example-chip').filter({ hasText: 'Chat:' }).click()   // auto-generates
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })
    await $('btn-live').click()
    // crop to the app column, from the header down to the run summary
    const head = await page.locator('.app-header').boundingBox()
    const end = await $('detail-run-end').boundingBox()
    const pad = 16
    await page.screenshot({ path: `${DIR}/pipeline.png`, clip: {
      x: head.x - pad, y: Math.max(0, head.y - pad), width: head.width + 2 * pad, height: end.y + end.height - head.y + 2 * pad,
    } })
    await $('temp-input').fill('0.8')
    console.log('pipeline.png')
  }
  // the remaining scenes continue text, as they did before Chat mode existed
  if (await $('format-text').count()) await $('format-text').click()

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

  // ---- logits-readout.png: real mode, Logits stage — equation row + bench
  if (want('logits')) {
    await $('prompt-input').fill('The farmer bought a horse and rode it home.')
    await $('maxtok-input').fill('1')                               // one cycle: the first token's distribution is deterministic
    const before = await $('run-chip').count()
    await $('btn-generate').click()
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })   // wait for the run to SEAL
    await $('btn-live').click()
    await $('stage-card').nth(3).click()                            // Logits
    await $('readout-bench').waitFor({ timeout: 10000 })
    await $('readout-candidate').nth(1).click()                     // second candidate on the bench
    await $('detail-logits').scrollIntoViewIfNeeded()
    await page.waitForTimeout(300)
    const box = await $('detail-logits').boundingBox()
    const pad = 12
    await page.screenshot({ path: `${DIR}/logits-readout.png`, clip: {
      x: box.x - pad, y: box.y - pad, width: box.width + 2 * pad, height: box.height + 2 * pad,
    } })
    console.log('logits-readout.png')
    // ---- logits-softmax.png: the next stage — the softmax step explained
    await $('stage-card').nth(4).click()                            // Softmax
    await $('softmax-explainer').waitFor({ timeout: 5000 })
    await page.setViewportSize({ width: 1380, height: 1400 })       // the explained card is taller than the default viewport
    await $('detail-softmax').scrollIntoViewIfNeeded()
    await page.waitForTimeout(300)
    const sbox = await $('detail-softmax').boundingBox()
    await page.screenshot({ path: `${DIR}/logits-softmax.png`, clip: {
      x: sbox.x - pad, y: sbox.y - pad, width: sbox.width + 2 * pad, height: sbox.height + 2 * pad,
    } })
    await page.setViewportSize({ width: 1380, height: 940 })
    await $('maxtok-input').fill('20')
    console.log('logits-softmax.png')
  }

  // ---- mlp.png: real mode, Layers stage — edit sizes and per-layer votes
  if (want('mlp')) {
    await $('prompt-input').fill('The farmer bought a horse and rode it home.')
    await $('maxtok-input').fill('1')
    const before = await $('run-chip').count()
    await $('btn-generate').click()
    await $('run-chip').nth(before).waitFor({ timeout: 120000 })
    await $('btn-live').click()
    await $('stage-card').nth(2).click()                            // Layers
    await $('vote-cell').first().waitFor({ timeout: 10000 })
    // the cards area also holds Inside one MLP and Step by step: hide those two;
    // the sticky playback and section bars would cover the cards once scrolled there
    const crop = await page.addStyleTag({ content: '#layers-neurons, #layers-steps { display: none !important; } '
      + '.controls, .layers-nav { position: static !important; }' })
    await page.waitForTimeout(300)
    await $('mlp-cards').screenshot({ path: `${DIR}/mlp.png` })
    await crop.evaluate((el) => el.remove())
    console.log('mlp.png')
    // ---- mlp-steps.png: the same run, The MLP, step by step -------------------
    const unstick = await page.addStyleTag({ content: '.controls, .layers-nav { position: static !important; }' })
    await $('steps-card').scrollIntoViewIfNeeded()
    await page.locator('[data-testid="steps-card"] svg').first().waitFor({ timeout: 60000 })
    // steps 1 and 5 need the layer's weights, fetched once per model version
    await page.waitForFunction(() => !document.querySelector('[data-testid="steps-loading"]'), null, { timeout: 60000 })
    await page.waitForTimeout(500)
    await $('steps-card').screenshot({ path: `${DIR}/mlp-steps.png` })
    await unstick.evaluate((el) => el.remove())
    console.log('mlp-steps.png')
    await $('maxtok-input').fill('20')
  }

  // ---- neurons.png: real mode, Layers stage — Inside one MLP ----------------
  if (want('neurons')) {
    await $('prompt-input').fill('The cat sat on the mat because it was tired')
    // three tokens and the last one viewed. The 8-bit model's input rounding
    // leaves some layers at "≈" (#19), often the default layer 29: regenerate
    // (sampling varies) until the shown total is exact, up to 4 attempts
    await $('maxtok-input').fill('3')
    for (let attempt = 0; attempt < 4; attempt++) {
      const before = await $('run-chip').count()
      await $('btn-generate').click()
      await $('run-chip').nth(before).waitFor({ timeout: 120000 })
      await $('btn-live').click()
      await $('stage-card').nth(2).click()                          // Layers
      await $('neuron-total').waitFor({ timeout: 30000 })
      if (!(await $('neuron-total').textContent()).includes('≈')) break
    }
    const unstick = await page.addStyleTag({ content: '.controls { position: static !important; }' })
    await page.waitForTimeout(300)
    await $('neuron-card').screenshot({ path: `${DIR}/neurons.png` })
    await unstick.evaluate((el) => el.remove())
    await $('maxtok-input').fill('20')
    console.log('neurons.png')
  }
} finally {
  await context.close()
  stopServer()
}

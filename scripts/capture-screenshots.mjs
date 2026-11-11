// Regenerates docs/screenshots/*.png against the current build.
//
//   npm run screenshots
//
// Leaves runs that were already in the profile as they were and removes the ones it makes.
// The archive keeps only 8 unpinned runs and evicts the oldest, and this script makes more than
// that, so it pins every unpinned run it finds first and unpins exactly those at the end. If the
// saved runs cannot be read or do not show up in the page, it stops before touching anything.
// If it aborts after the first pin (error, Ctrl-C, SIGTERM), the same cleanup runs: runs made here
// are removed and runs pinned here are unpinned.
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
  colorScheme: 'light',   // every shot is light except dark.png, whatever this computer is set to
})
// Scenes can be selected: `npm run screenshots -- glossary pipeline attention compare embeddings logits mlp neurons knockouts`
// (mlp also writes mlp-steps.png; attention also writes head-explorer.png and dark.png)
// (default: all). Skipping a scene preserves its existing images — useful when
// a previous capture caught a specimen worth keeping (e.g. a great fork).
const only = process.argv.slice(2)
const want = (scene) => only.length === 0 || only.includes(scene)

let restoreTheme = async () => {}  // set once the page is open
let cleanup = null   // armed just before the first run is pinned; run when a scene fails or on Ctrl-C / SIGTERM
let cleaning = null  // the cleanup in progress, so it never runs twice
const runCleanup = () => (cleaning ??= cleanup())
for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, async () => {
    if (cleanup) await runCleanup().catch((e) => console.error('cleanup failed:', e))
    stopServer()
    process.exit(130)
  })
}
try {
  await waitForServer()
  const page = context.pages()[0] ?? await context.newPage()
  await page.goto(URL)
  // the shots need the theme on Auto, so the emulated system setting decides; a stored
  // choice in this profile is put back at the end
  const storedTheme = await page.evaluate(() => localStorage.getItem('tsumugi.theme'))
  if (storedTheme !== null) {
    await page.evaluate(() => localStorage.removeItem('tsumugi.theme'))
    await page.reload()
  }
  restoreTheme = async () => {
    if (storedTheme !== null) await page.evaluate((v) => localStorage.setItem('tsumugi.theme', v), storedTheme)
  }
  const $ = (id) => page.getByTestId(id)
  // before every shot: the pointer off the controls and the 0.15 to 0.3 s colour transitions finished, so no
  // picture shows a hovered button or a colour caught half way
  const settle = async () => { await page.mouse.move(2, 2); await page.waitForTimeout(400) }

  // The profile persists IndexedDB, so runs from earlier sessions survive reloads. The shelf shows only the newest
  // five runs plus the active one, so runs are told apart by their number (the newest chip is first), not counted.
  const topNum = async () => {
    const t = await $('run-chip').first().textContent({ timeout: 1000 }).catch(() => null)
    return t ? Number(t.match(/#(\d+)/)?.[1]) : 0
  }
  // Read the saved runs straight from IndexedDB, so "no runs" can never be a guess made from a slow page.
  const saved = await page.evaluate(() => new Promise((resolve, reject) => {
    const req = indexedDB.open('tsumugi', 1)
    req.onupgradeneeded = () => { req.result.createObjectStore('runs', { keyPath: 'id' }) }
    req.onerror = () => reject(req.error)
    req.onsuccess = () => {
      const all = req.result.transaction('runs', 'readonly').objectStore('runs').getAll()
      all.onerror = () => reject(all.error)
      all.onsuccess = () => { req.result.close(); resolve(all.result.map((r) => ({ seq: r.meta.seq, pinned: r.meta.pinned }))) }
    }
  })).catch((err) => { throw new Error(`cannot read the saved runs (${err}); not touching the archive`) })
  const startNum = Math.max(0, ...saved.map((r) => r.seq))
  if (saved.length > 0) {
    // the page has loaded the archive once its newest chip is the newest saved run
    await page.waitForFunction((n) => {
      const t = document.querySelector('[data-testid="run-chip"]')?.textContent
      return !!t && Number(t.match(/#(\d+)/)?.[1]) === n
    }, startNum, { timeout: 15000 }).catch(() => {
      throw new Error(`the page did not show saved run #${startNum} within 15 s; the archive state is unknown, so nothing was changed`)
    })
  }
  // a run's row: in the open "+N more" list when there is one, else its chip (hovered so its buttons show)
  const rowOf = async (seq) => {
    const label = new RegExp(`(?<!\\d)#${seq} ·`)
    if (await $('run-more').count() && !(await $('run-list').count())) await $('run-more').click()
    const inList = $('run-list-row').filter({ hasText: label })
    if (await inList.count()) return inList
    const chip = $('run-chip').filter({ hasText: label })
    await chip.hover()
    return chip
  }
  const closeList = async () => { if (await $('run-list').count()) await page.keyboard.press('Escape') }
  const pinnedByScript = []   // filled as each run is pinned, so an abort part-way unpins exactly those
  // remove only the runs made here, then unpin the ones pinned at the start
  const madeHere = async () => {
    const nums = new Set()
    for (const t of await $('run-chip').allTextContents()) nums.add(Number(t.match(/#(\d+)/)?.[1]))
    if (await $('run-more').count()) {
      await $('run-more').click()
      for (const t of await $('run-list-row').allTextContents()) nums.add(Number(t.match(/#(\d+)/)?.[1]))
      await closeList()
    }
    return [...nums].filter((n) => n > startNum)
  }
  // armed before the first pin, so any later abort (error, Ctrl-C) undoes what has been done so far
  cleanup = async () => {
    if (await $('btn-compare-exit').count()) await $('btn-compare-exit').click()
    for (let left = await madeHere(); left.length > 0; left = await madeHere()) {
      const row = await rowOf(left[0])
      await row.getByTestId('btn-chip-remove').click()
      await closeList()
    }
    while (pinnedByScript.length > 0) {
      const row = await rowOf(pinnedByScript[0])
      await row.getByTestId('btn-chip-pin').click()
      await closeList()
      pinnedByScript.shift()
    }
  }
  for (const { seq } of saved.filter((r) => !r.pinned)) {
    const row = await rowOf(seq)
    await row.getByTestId('btn-chip-pin').click()
    pinnedByScript.push(seq)
    await closeList()
  }
  const newerThan = (n) => page.waitForFunction((n) => {
    const t = document.querySelector('[data-testid="run-chip"]')?.textContent
    return t && Number(t.match(/#(\d+)/)[1]) > n
  }, n, { timeout: 120000 })
  // the examples row starts closed once runs are saved
  const openExamples = async () => { if (!(await $('example-chip').count())) await $('toggle-examples').click() }

  // ---- glossary.png: simulated run, Softmax card with the drawer open -------
  if (want('glossary')) {
    const prev = await topNum()
    await openExamples()
    await $('example-chip').first().click()                  // auto-generates
    await newerThan(prev)
    await $('btn-live').click()
    await $('stage-card').nth(4).click()                     // Softmax
    await $('embed-callout').filter({ hasText: 'why exponentiate' }).click()
    await $('gl-in-run').waitFor({ timeout: 5000 })
    await page.waitForTimeout(800)                           // let the section scroll and the highlight settle
    await settle()
    await page.screenshot({ path: `${DIR}/glossary.png` })
    await $('gl-close').click()
    console.log('glossary.png')
  }

  // real mode serves the remaining scenes
  if (want('pipeline') || want('attention') || want('compare') || want('embeddings') || want('logits') || want('mlp') || want('neurons') || want('knockouts')) {
    await $('mode-real').click()
    await $('mode-badge').filter({ hasText: 'attn' }).waitFor({ timeout: 300000 })  // first run downloads the model
  }

  // ---- pipeline.png: real model, the chat example at run end ----------------
  if (want('pipeline')) {
    await $('temp-input').fill('0')                          // greedy: the answer ends its turn with <|im_end|>
    const before = await topNum()
    await openExamples()
    await $('example-chip').filter({ hasText: 'Chat:' }).click()   // auto-generates
    await newerThan(before)
    await $('btn-live').click()
    // crop to the app column, from the header down to the run summary
    const head = await page.locator('.app-header').boundingBox()
    const end = await $('detail-run-end').boundingBox()
    const pad = 16
    await settle()
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
    const before = await topNum()
    await openExamples()
    await $('example-chip').first().click()
    await newerThan(before)
    await $('btn-live').click()
    await $('stage-card').nth(2).click()                     // Layers
    await $('detail-layers').scrollIntoViewIfNeeded()
    await settle()
    await page.screenshot({ path: `${DIR}/real-attention.png` })
    console.log('real-attention.png')
    // dark.png: the same view with the system set to dark (the theme is on Auto)
    await page.emulateMedia({ colorScheme: 'dark' })
    await page.waitForTimeout(400)
    await settle()
    await page.screenshot({ path: `${DIR}/dark.png` })
    await page.emulateMedia({ colorScheme: 'light' })
    await page.waitForTimeout(400)
    console.log('dark.png')
    await $('btn-explore-heads').click()
    await $('grid-explorer').scrollIntoViewIfNeeded()
    await page.waitForTimeout(500)                           // let the canvases paint
    await settle()
    await page.screenshot({ path: `${DIR}/head-explorer.png` })
    console.log('head-explorer.png')
  }

  // ---- compare.png + compare-attention.png: same-prompt fork ----------------
  if (want('compare')) {
  await $('prompt-input').fill('The cat sat on the mat because it was tired')
  await $('temp-input').fill('0.9')
  {
    const before = await topNum()               // baseline T=0.9 run,
    await $('btn-generate').click()                          // so the pair shares T
    await newerThan(before)
  }
  for (let attempt = 0; attempt < 4; attempt++) {
    const before = await topNum()
    await $('btn-generate').click()                          // same prompt again
    await newerThan(before)  // wait for the run to SEAL
    await $('btn-compare-arm').click()                       // in the header now
    await page.locator('[data-testid="run-chip-main"]', { hasText: new RegExp(`#${before} `) }).click()   // B = previous run
    await $('compare-view').waitFor({ timeout: 5000 })
    if (await page.locator('[data-testid="cmp-tick"][data-fork="true"]').count()) break
    console.log('no fork this time — regenerating')
    await $('btn-compare-exit').click()
  }
  const fork = page.locator('[data-testid="cmp-tick"][data-fork="true"]')
  await ((await fork.count()) ? fork.first() : $('cmp-tick').nth(2)).click()
  await $('cmp-dist-side').first().waitFor({ timeout: 5000 })
  await page.evaluate(() => window.scrollTo(0, 0))
  await settle()
  await page.screenshot({ path: `${DIR}/compare.png` })
  console.log('compare.png')
  await $('cmp-attn-side').first().scrollIntoViewIfNeeded()
  await settle()
  await page.screenshot({ path: `${DIR}/compare-attention.png` })
  console.log('compare-attention.png')
  }

  // ---- embeddings.png: real mode, Embeddings stage — lookup + neighbours + matrix
  if (want('embeddings')) {
    await $('prompt-input').fill('The cat sat on the mat because the cat was tired')
    await $('maxtok-input').fill('8')                               // short run: the card stays one screen tall
    const before = await topNum()
    await $('btn-generate').click()
    await newerThan(before)   // wait for the run to SEAL
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
    await settle()
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
    const before = await topNum()
    await $('btn-generate').click()
    await newerThan(before)   // wait for the run to SEAL
    await $('btn-live').click()
    await $('stage-card').nth(3).click()                            // Logits
    await $('readout-bench').waitFor({ timeout: 10000 })
    await $('readout-candidate').nth(1).click()                     // second candidate on the bench
    await $('detail-logits').scrollIntoViewIfNeeded()
    await page.waitForTimeout(300)
    const box = await $('detail-logits').boundingBox()
    const pad = 12
    await settle()
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
    await settle()
    await page.screenshot({ path: `${DIR}/logits-softmax.png`, clip: {
      x: sbox.x - pad, y: sbox.y - pad, width: sbox.width + 2 * pad, height: sbox.height + 2 * pad,
    } })
    await page.setViewportSize({ width: 1380, height: 940 })
    await $('maxtok-input').fill('20')
    console.log('logits-softmax.png')
  }

  // ---- knockouts.png: real mode, Layers stage — one sweep, one layer picked
  if (want('knockouts')) {
    await $('temp-input').fill('0')                                 // greedy, so the answer token is stable
    await $('prompt-input').fill('The farmer bought a horse and rode it home.')
    await $('maxtok-input').fill('1')
    const before = await topNum()
    await $('btn-generate').click()
    await newerThan(before)
    await $('btn-live').click()
    await $('stage-card').nth(2).click()                            // Layers
    await $('knockout-card').waitFor({ timeout: 10000 })
    await page.getByRole('button', { name: 'Measure all 30 layers' }).click()
    await page.waitForFunction(() => document.querySelectorAll('[data-testid="knockout-bar"]').length === 30, null, { timeout: 120000 })
    await $('knockout-chip').first().waitFor({ timeout: 10000 })
    await $('knockout-chip').first().click()                        // layer 0 (breaks the model) sorts last: detail line + run button
    await $('knockout-detail').waitFor({ timeout: 5000 })
    const unstick = await page.addStyleTag({ content: '.controls, .layers-nav { position: static !important; }' })
    await $('knockout-card').scrollIntoViewIfNeeded()
    await page.waitForTimeout(300)
    await settle()
    await $('knockout-card').screenshot({ path: `${DIR}/knockouts.png` })
    await unstick.evaluate((el) => el.remove())
    await $('temp-input').fill('0.8')
    await $('maxtok-input').fill('20')
    console.log('knockouts.png')
  }

  // ---- mlp.png: real mode, Layers stage — edit sizes and per-layer votes
  if (want('mlp')) {
    await $('prompt-input').fill('The farmer bought a horse and rode it home.')
    await $('maxtok-input').fill('1')
    const before = await topNum()
    await $('btn-generate').click()
    await newerThan(before)
    await $('btn-live').click()
    await $('stage-card').nth(2).click()                            // Layers
    await $('vote-cell').first().waitFor({ timeout: 10000 })
    // the cards area also holds Inside one MLP and Step by step: hide those two;
    // the sticky playback and section bars would cover the cards once scrolled there
    const crop = await page.addStyleTag({ content: '#layers-neurons, #layers-steps { display: none !important; } '
      + '.controls, .layers-nav { position: static !important; }' })
    await page.waitForTimeout(300)
    await settle()
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
    await settle()
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
      const before = await topNum()
      await $('btn-generate').click()
      await newerThan(before)
      await $('btn-live').click()
      await $('stage-card').nth(2).click()                          // Layers
      await $('neuron-total').waitFor({ timeout: 30000 })
      if (!(await $('neuron-total').textContent()).includes('≈')) break
    }
    const unstick = await page.addStyleTag({ content: '.controls { position: static !important; }' })
    await page.waitForTimeout(300)
    await settle()
    await $('neuron-card').screenshot({ path: `${DIR}/neurons.png` })
    await unstick.evaluate((el) => el.remove())
    await $('maxtok-input').fill('20')
    console.log('neurons.png')
  }
  await runCleanup()
  cleanup = null
} catch (err) {
  if (cleanup) await runCleanup().catch((e) => console.error('cleanup failed:', e))
  throw err
} finally {
  await restoreTheme().catch(() => { /* the page may already be gone */ })
  await context.close()
  stopServer()
}

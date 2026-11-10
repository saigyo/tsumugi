// hover + contrast measurements with the page in the dark theme: node dark.mjs sim|real
import { rmSync } from 'node:fs'
import { chromium } from '/Users/markus/IdeaProjects/llm-visualizer/node_modules/@playwright/test/index.mjs'
import { startServer, installPageFns, makeTools, URL } from './lib.mjs'
const mode = process.argv[2] ?? 'sim'
const BASE = new globalThis.URL('.', import.meta.url).pathname
setTimeout(() => { console.log('HARD TIMEOUT'); process.exit(2) }, 14 * 60 * 1000)
const stop = await startServer()
const PROF = mode === 'sim' ? `${BASE}prof-dark` : `${BASE}prof-clone`
if (mode === 'sim') rmSync(PROF, { recursive: true, force: true })
const context = await chromium.launchPersistentContext(PROF, { headless: mode === 'sim', viewport: { width: 1380, height: 940 }, colorScheme: 'dark' })
const page = context.pages()[0] ?? await context.newPage()
await installPageFns(page)
const T = makeTools(page, `${BASE}out-dark-${mode}`)
const { $, log, scene, topNum, newerThan, openExamples } = T
const m = async (label, scope = 'body') => { await page.waitForTimeout(700); await T.metrics(label); await T.hoverAudit(label, scope) }
try {
  await page.goto(URL); await page.evaluate(() => localStorage.removeItem('tsumugi.theme')); await page.reload(); await page.waitForTimeout(1200)
  log('theme', await page.evaluate(() => document.documentElement.dataset.theme))
  if (mode === 'sim') {
    await scene('first', async () => { await m('first-visit') })
    await scene('run', async () => {
      await $('example-chip').first().click(); await newerThan(0, 60000)
      await $('btn-live').click({ timeout: 3000 }).catch(() => {}); await m('run-end')
      for (const [i, nm] of ['tokenizer', 'embeddings', 'layers', 'logits', 'softmax', 'sampler'].entries()) {
        if (i === 0) continue
        await $('stage-card').nth(i).click(); await m(`stage-${nm}`)
      }
      await $('generated-token').nth(1).hover({ timeout: 3000 }); await page.waitForTimeout(300); await T.metrics('popover'); await page.mouse.move(5, 5)
      const term = page.locator('.detail .gl-term').first()
      if (await term.count()) { await term.hover(); await page.waitForTimeout(400); await T.metrics('term-tip'); await page.mouse.move(5, 5) }
    })
    await scene('glossary', async () => {
      await $('btn-help').first().click(); await $('glossary-drawer').waitFor({ timeout: 5000 }); await m('gl-index', '.gl-drawer')
      const ids = await $('gl-index-row').evaluateAll((els) => els.map((e) => e.getAttribute('data-entry')))
      for (const id of ids) {
        await page.locator(`[data-testid="gl-index-row"][data-entry="${id}"]`).click({ timeout: 3000 }); await page.waitForTimeout(800); await T.metrics(`gl-${id}`)
        await $('gl-all-entries').click({ timeout: 3000 }).catch(() => {})
      }
      await $('gl-close').click()
    })
    await scene('runs', async () => {
      await page.evaluate(() => scrollTo(0, 0))
      for (const p of ['Once upon a time', 'Paris is the capital of', 'Hello', 'The farmer bought a horse', '2 + 2 =', 'One more', 'And another']) {
        const before = await topNum(); await $('prompt-input').fill(p); await $('btn-generate').click(); await newerThan(before, 60000)
      }
      await $('run-chip').nth(2).getByTestId('btn-chip-pin').click().catch(() => {}); await page.mouse.move(5, 5)
      await m('runs', '.run-shelf')
      await $('run-more').click(); await m('runs-list', '.run-list'); await page.keyboard.press('Escape')
      await $('btn-compare-arm').click(); await m('compare-armed', '.run-shelf')
      await $('run-chip-main').nth(1).click(); await $('compare-view').waitFor({ timeout: 5000 })
      await $('cmp-tick').nth(2).click().catch(() => {}); await page.mouse.move(5, 5); await m('compare', '.compare-view')
      if (await $('dist-mode-combined').count()) { await $('dist-mode-combined').click(); await page.mouse.move(5, 5); await m('compare-combined', '.compare-view') }
      await $('btn-compare-exit').click()
    })
  } else {
    await scene('real', async () => {
      await $('mode-real').click(); await $('mode-badge').filter({ hasText: 'attn' }).waitFor({ timeout: 300000 })
      await $('format-chat').click().catch(() => {})
      await $('temp-input').fill('0'); await $('maxtok-input').fill('20'); await openExamples(); await m('real-top', '.prompt-box')
      const prev = await topNum(); await $('example-chip').filter({ hasText: 'Chat:' }).click(); await newerThan(prev); await page.waitForTimeout(6000)
      await m('real-runend'); await $('btn-live').click()
      for (const [i, nm] of ['tokenizer', 'embeddings', 'layers', 'logits', 'softmax', 'sampler'].entries()) {
        await $('stage-card').nth(i).click(); await page.waitForTimeout(1500)
        if (nm === 'embeddings') await $('embed-neighbors').waitFor({ timeout: 60000 }).catch(() => {})
        if (nm === 'layers') { await $('vote-cell').first().waitFor({ timeout: 20000 }); await $('steps-card').scrollIntoViewIfNeeded(); await page.waitForFunction(() => !document.querySelector('[data-testid="steps-loading"]'), null, { timeout: 60000 }).catch(() => {}); await page.waitForTimeout(1200) }
        await m(`real-${nm}`, '.detail')
      }
      await $('stage-card').nth(2).click(); await $('knockout-card').waitFor({ timeout: 10000 })
      await page.getByRole('button', { name: 'Measure all 30 layers' }).click()
      await page.waitForFunction(() => document.querySelectorAll('[data-testid="knockout-bar"]').length === 30, null, { timeout: 120000 })
      await $('knockout-chip').first().click(); await $('knockout-detail').waitFor({ timeout: 5000 }); await page.mouse.move(5, 5)
      await m('knockout', '[data-testid="knockout-card"]')
      await $('btn-explore-heads').click(); await $('grid-explorer').waitFor({ timeout: 10000 }); await m('explorer', '.grid-explorer')
      const dot = page.locator('[data-testid="steps-card"] .step-dot-marked').first()
      if (await dot.count()) { await dot.scrollIntoViewIfNeeded(); await dot.hover({ force: true }); await page.waitForTimeout(400); await T.metrics('steps-hover') }
    })
  }
  log('DONE')
} catch (e) { console.error(e) } finally { await context.close(); stop(); process.exit(0) }

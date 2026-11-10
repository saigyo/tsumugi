// real-model pass: a clone of .screenshots-profile (model cached), headed for WebGPU
import { chromium } from '/Users/markus/IdeaProjects/llm-visualizer/node_modules/@playwright/test/index.mjs'
import { startServer, installPageFns, makeTools, URL } from './lib.mjs'

const BASE = new globalThis.URL('.', import.meta.url).pathname
const OUT = `${BASE}out-real`
const PROF = `${BASE}prof-clone`
setTimeout(() => { console.log('HARD TIMEOUT'); process.exit(2) }, 28 * 60 * 1000)
const stop = await startServer()
const context = await chromium.launchPersistentContext(PROF, { headless: false, viewport: { width: 1380, height: 940 }, colorScheme: 'light' })
const page = context.pages()[0] ?? await context.newPage()
await installPageFns(page)
const T = makeTools(page, OUT)
const { $, log, metrics, hoverAudit, shotEl, shotApp, shotView, scene, topNum, newerThan, openExamples, unstick } = T
const TOP = { x: 200, y: 0, width: 980, height: 560 }
try {
  await page.goto(URL)
  await page.evaluate(() => localStorage.removeItem('tsumugi.theme'))
  await page.reload()
  await page.waitForTimeout(1500)

  await scene('load', async () => {
    await shotView('20-returning-visitor', ['light'], TOP)
    await metrics('returning')
    await $('mode-real').click()
    await page.waitForTimeout(350)
    await shotView('20-model-loading', ['light', 'dark'], TOP)
    await metrics('model-loading')
    await $('mode-badge').filter({ hasText: 'attn' }).waitFor({ timeout: 300000 })
    await page.waitForTimeout(500)
    await shotView('20-real-ready', ['light', 'dark'], TOP)
    await metrics('real-ready')
    await hoverAudit('real-top', '.prompt-box')
  })

  await scene('chat', async () => {
    await $('temp-input').fill('0')
    await openExamples()
    await shotView('21-examples-open-real', ['light'], TOP)
    const prev = await topNum()
    await $('example-chip').filter({ hasText: 'Chat:' }).click()
    await page.waitForTimeout(1500)
    await shotView('21-generating-real', ['light', 'dark'], TOP)
    await newerThan(prev)
    await page.waitForTimeout(9000)
    await shotApp('22-real-runend')
    await metrics('real-runend')
    await hoverAudit('real-runend', '.detail')
    await $('toggle-system').click({ timeout: 3000 })
    await page.waitForTimeout(300)
    await shotView('22-system-open', ['light', 'dark'], TOP)
    await metrics('system-open')
    await $('toggle-system').click({ timeout: 3000 })
  })

  await scene('stages', async () => {
    await $('btn-live').click()
    const u = await unstick()
    for (const [i, nm] of ['tokenizer', 'embeddings', 'layers', 'logits', 'softmax', 'sampler'].entries()) {
      await scene(nm, async () => {
        await $('stage-card').nth(i).click()
        await page.waitForTimeout(1500)
        if (nm === 'embeddings') await $('embed-neighbors').waitFor({ timeout: 60000 }).catch(() => {})
        if (nm === 'layers') {
          await $('vote-cell').first().waitFor({ timeout: 20000 })
          await $('steps-card').scrollIntoViewIfNeeded()
          await page.waitForFunction(() => !document.querySelector('[data-testid="steps-loading"]'), null, { timeout: 60000 }).catch(() => {})
          await page.waitForTimeout(1500)
        }
        const top = await page.locator('.detail').first().boundingBox()
        const sy = await page.evaluate(() => scrollY)
        await shotApp(`23-stage-${i}-${nm}`, ['light', 'dark'], Math.max(0, top.y + sy - 10))
        await metrics(`real-${nm}`)
        await hoverAudit(`real-${nm}`, '.detail')
      })
    }
    await u.evaluate((el) => el.remove())
  })

  await scene('layers-detail', async () => {
    await $('stage-card').nth(2).click()
    await $('knockout-card').waitFor({ timeout: 10000 })
    // sticky bars while scrolled into the Layers detail
    await $('votes-card').first().scrollIntoViewIfNeeded()
    await page.mouse.wheel(0, 300); await page.waitForTimeout(500)
    await shotView('24-sticky-bars', ['light', 'dark'], { x: 200, y: 0, width: 980, height: 500 })
    await page.getByRole('button', { name: 'Measure all 30 layers' }).scrollIntoViewIfNeeded()
    await page.getByRole('button', { name: 'Measure all 30 layers' }).hover(); await page.waitForTimeout(300)
    await shotEl('24-knockout-before-hover', $('knockout-card'), ['light'])
    await page.getByRole('button', { name: 'Measure all 30 layers' }).click()
    await page.waitForTimeout(700)
    await shotEl('24-knockout-measuring', $('knockout-card'), ['light'])
    await page.waitForFunction(() => document.querySelectorAll('[data-testid="knockout-bar"]').length === 30, null, { timeout: 120000 })
    await $('knockout-chip').first().click()
    await $('knockout-detail').waitFor({ timeout: 5000 })
    await page.waitForTimeout(500)
    const u = await unstick()
    await shotEl('24-knockout-measured', $('knockout-card'))
    await metrics('knockout-measured')
    await $('knockout-bar').nth(12).hover(); await page.waitForTimeout(300)
    await shotEl('24-knockout-bar-hover', $('knockout-card'), ['light'])
    await hoverAudit('knockout', '[data-testid="knockout-card"]')
    await $('vote-cell').nth(40).hover(); await page.waitForTimeout(300)
    await shotEl('24-votes-hover', $('votes-card').first(), ['light'])
    await $('attention-heatmap').first().scrollIntoViewIfNeeded()
    await $('attn-cell').nth(5).hover().catch(() => {})
    await page.waitForTimeout(300)
    await shotEl('24-heatmap-hover', $('attention-heatmap').first(), ['light'])
    await $('btn-explore-heads').click()
    await $('grid-explorer').waitFor({ timeout: 10000 })
    await page.waitForTimeout(1200)
    await shotEl('24-explorer', $('grid-explorer'))
    await $('grid-cell').nth(7).hover(); await page.waitForTimeout(300)
    await shotEl('24-explorer-hover', $('grid-explorer'), ['light'])
    await metrics('explorer')
    await hoverAudit('explorer', '.grid-explorer')
    // step-box tooltip in the steps card
    const dot = page.locator('[data-testid="steps-card"] .step-dot-marked').first()
    if (await dot.count()) { await dot.scrollIntoViewIfNeeded(); await dot.hover({ force: true }); await page.waitForTimeout(400); await shotEl('24-steps-hover', $('steps-card')) }
    await u.evaluate((el) => el.remove())
  })

  await scene('narrow-real', async () => {
    for (const w of [1024, 768, 400]) {
      await page.setViewportSize({ width: w, height: 900 }); await page.waitForTimeout(600)
      const u = await unstick()
      const top = await page.locator('.app-header').boundingBox()
      await page.evaluate(() => scrollTo(0, 0))
      if (w !== 1024) await shotApp(`25-real-layers-w${w}`, ['light'])
      await metrics(`real-layers-w${w}`)
      await u.evaluate((el) => el.remove())
      await page.evaluate(() => scrollTo(0, 0))
      await shotView(`25-real-top-w${w}`, ['light'], { x: 0, y: 0, width: w, height: 620 })
    }
    await page.setViewportSize({ width: 1380, height: 940 })
  })

  await scene('compare', async () => {
    await page.evaluate(() => scrollTo(0, 0))
    await $('format-text').click()
    await $('temp-input').fill('0.9')
    await $('prompt-input').fill('The cat sat on the mat because it was tired')
    await $('maxtok-input').fill('8')
    let before = await topNum()
    await $('btn-generate').click(); await newerThan(before)
    for (let a = 0; a < 3; a++) {
      before = await topNum()
      await $('btn-generate').click(); await newerThan(before)
      await $('btn-compare-arm').click()
      await page.locator('[data-testid="run-chip-main"]', { hasText: new RegExp(`#${before} `) }).click()
      await $('compare-view').waitFor({ timeout: 5000 })
      if (await page.locator('[data-testid="cmp-tick"][data-fork="true"]').count()) break
      await $('btn-compare-exit').click()
    }
    const fork = page.locator('[data-testid="cmp-tick"][data-fork="true"]')
    await ((await fork.count()) ? fork.first() : $('cmp-tick').nth(2)).click()
    await $('cmp-dist-side').first().waitFor({ timeout: 5000 })
    await page.waitForTimeout(800)
    await shotApp('26-compare-real')
    await metrics('compare-real')
    await hoverAudit('compare-real', '.compare-view')
    await $('btn-compare-exit').click()
    await page.waitForTimeout(300)
    await shotView('26-runs-after', ['light', 'dark'], TOP)
  })
  log('DONE')
} catch (e) { console.error(e) } finally { await context.close(); stop(); process.exit(0) }

// reduced sim pass in another engine: node cross.mjs webkit
import * as pw from '/Users/markus/IdeaProjects/llm-visualizer/node_modules/@playwright/test/index.mjs'
import { startServer, installPageFns, makeTools, URL } from './lib.mjs'
const engine = process.argv[2] ?? 'webkit'
const BASE = new globalThis.URL('.', import.meta.url).pathname
setTimeout(() => { console.log('HARD TIMEOUT'); process.exit(2) }, 8 * 60 * 1000)
const stop = await startServer()
const browser = await pw[engine].launch()
const context = await browser.newContext({ viewport: { width: 1380, height: 940 }, colorScheme: 'light' })
const page = await context.newPage()
const errors = []
page.on('pageerror', (e) => errors.push(String(e).slice(0, 200)))
page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text().slice(0, 200)) })
await installPageFns(page)
const T = makeTools(page, `${BASE}out-${engine}`)
const { $, log, metrics, hoverAudit, shotEl, shotApp, shotView, scene, topNum, newerThan, unstick } = T
try {
  await page.goto(URL); await page.waitForTimeout(1000)
  await scene('first', async () => { await shotApp('01-first-visit'); await metrics('first-visit') })
  await scene('run', async () => {
    await $('example-chip').first().click(); await newerThan(0, 60000)
    await $('btn-live').click({ timeout: 3000 }).catch(() => {})
    await page.waitForTimeout(500)
    await shotApp('03-run-end'); await metrics('run-end'); await hoverAudit('run-end')
  })
  await scene('stages', async () => {
    const u = await unstick()
    for (const [i, nm] of ['tokenizer', 'embeddings', 'layers', 'logits', 'softmax', 'sampler'].entries()) {
      if (i === 0) continue
      await $('stage-card').nth(i).click(); await page.waitForTimeout(700)
      const top = await page.locator('.token-stream').boundingBox(); const sy = await page.evaluate(() => scrollY)
      await shotApp(`04-stage-${i}-${nm}`, nm === 'layers' ? ['light', 'dark'] : ['light'], Math.max(0, top.y + sy - 10))
      await metrics(`stage-${nm}`)
    }
    await u.evaluate((el) => el.remove())
  })
  await scene('popover', async () => {
    await page.evaluate(() => scrollTo(0, 0))
    await $('generated-token').nth(1).hover({ timeout: 3000 }); await page.waitForTimeout(300)
    await shotView('05-popover', ['light'], { x: 200, y: 180, width: 980, height: 420 })
    await page.mouse.move(5, 5)
    await page.locator('.settings-line .gl-term-icon').first().hover(); await page.waitForTimeout(400)
    await shotView('05-icon-tip', ['light'], { x: 200, y: 60, width: 980, height: 300 })
    await page.mouse.move(5, 5)
    await $('temp-input').focus(); await page.waitForTimeout(200)
    await shotView('05-input-focus', ['light'], { x: 200, y: 60, width: 980, height: 300 })
  })
  await scene('glossary', async () => {
    await page.setViewportSize({ width: 1380, height: 1900 })
    await $('btn-help').first().click(); await $('glossary-drawer').waitFor({ timeout: 5000 }); await page.waitForTimeout(400)
    for (const id of ['mlp', 'softmax', 'votes']) {
      await page.locator(`[data-testid="gl-index-row"][data-entry="${id}"]`).click({ timeout: 3000 }); await page.waitForTimeout(400)
      await shotEl(`08-gl-${id}`, $('glossary-drawer'), ['light'], 0)
      await $('gl-all-entries').click({ timeout: 3000 }).catch(() => {})
    }
    await $('gl-close').click(); await page.setViewportSize({ width: 1380, height: 940 })
  })
  await scene('runs', async () => {
    await page.evaluate(() => scrollTo(0, 0))
    for (const p of ['Once upon a time', 'Paris is the capital of', 'Hello', 'The farmer bought a horse', '2 + 2 =', 'One more run', 'And another']) {
      const before = await topNum(); await $('prompt-input').fill(p); await $('btn-generate').click(); await newerThan(before, 60000)
    }
    const clip = { x: 200, y: 0, width: 980, height: 560 }
    await $('run-more').click(); await page.waitForTimeout(300)
    await shotView('09-runs-list', ['light', 'dark'], clip)
    await page.keyboard.press('Escape')
    await $('btn-compare-arm').click(); await $('run-chip-main').nth(1).click(); await $('compare-view').waitFor({ timeout: 5000 })
    await $('cmp-tick').nth(2).click().catch(() => {}); await page.mouse.move(5, 5); await page.waitForTimeout(500)
    await shotApp('10-compare', ['light']); await metrics('compare')
    await $('btn-compare-exit').click()
  })
  await scene('narrow', async () => {
    await page.setViewportSize({ width: 400, height: 900 }); await page.waitForTimeout(300)
    await $('stage-card').nth(4).click().catch(() => {}); await page.waitForTimeout(500)
    await shotApp('12-w400', ['light']); await metrics('w400')
  })
  log('ERRORS', JSON.stringify([...new Set(errors)].slice(0, 10)))
  log('DONE')
} catch (e) { console.error(e) } finally { await browser.close(); stop(); process.exit(0) }

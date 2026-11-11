// simulated-mode pass: fresh profile, headless
import { rmSync, writeFileSync } from 'node:fs'
import { chromium } from '/Users/markus/IdeaProjects/llm-visualizer/node_modules/@playwright/test/index.mjs'
import { startServer, installPageFns, makeTools, URL } from './lib.mjs'

const BASE = new globalThis.URL('.', import.meta.url).pathname
const OUT = `${BASE}out-sim`
const PROF = `${BASE}prof-fresh`
rmSync(PROF, { recursive: true, force: true })
setTimeout(() => { console.log('HARD TIMEOUT'); process.exit(2) }, 14 * 60 * 1000)
const stop = await startServer()
const context = await chromium.launchPersistentContext(PROF, { headless: true, viewport: { width: 1380, height: 940 }, colorScheme: 'light' })
await context.grantPermissions(['clipboard-read', 'clipboard-write']).catch(() => {})
const page = context.pages()[0] ?? await context.newPage()
await installPageFns(page)
const T = makeTools(page, OUT)
const { $, log, metrics, hoverAudit, shotEl, shotApp, shotView, scene, topNum, newerThan, unstick } = T
try {
  await page.goto(URL)
  await page.waitForTimeout(800)

  await scene('first-visit', async () => {
    await shotApp('01-first-visit')
    await metrics('first-visit')
    await hoverAudit('first-visit')
    for (const w of [1024, 768, 400]) {
      await page.setViewportSize({ width: w, height: 900 }); await page.waitForTimeout(300)
      await shotApp(`01-first-visit-w${w}`, w === 400 ? ['light', 'dark'] : ['light'])
      await metrics(`first-visit-w${w}`)
    }
    await page.setViewportSize({ width: 1380, height: 940 })
  })

  await scene('generate', async () => {
    await $('example-chip').first().click()
    await page.waitForTimeout(120)
    await shotView('02-generating', ['light', 'dark'], { x: 200, y: 0, width: 980, height: 520 })
    await metrics('generating')
    await newerThan(0, 60000)
    await page.waitForTimeout(600)
    await shotView('02b-after-seal-playing', ['light'], { x: 200, y: 0, width: 980, height: 940 })
    await $('btn-live').click({ timeout: 3000 }).catch(() => {})
    await page.waitForTimeout(500)
    await shotApp('03-run-end')
    await metrics('run-end')
    await hoverAudit('run-end')
  })

  await scene('stages', async () => {
    const u = await unstick()
    for (const [i, nm] of ['tokenizer', 'embeddings', 'layers', 'logits', 'softmax', 'sampler'].entries()) {
      await $('stage-card').nth(i).click()
      await page.waitForTimeout(700)
      const top = await page.locator('.token-stream').boundingBox()
      const sy = await page.evaluate(() => scrollY)
      await shotApp(`04-stage-${i}-${nm}`, ['light', 'dark'], Math.max(0, top.y + sy - 10))
      await metrics(`stage-${nm}`)
      await hoverAudit(`stage-${nm}`, '.detail')
    }
    await u.evaluate((el) => el.remove())
  })

  await scene('popover', async () => {
    await page.evaluate(() => scrollTo(0, 0))
    await $('generated-token').nth(1).hover({ timeout: 3000 })
    await page.waitForTimeout(300)
    await shotView('05-popover', ['light', 'dark'], { x: 200, y: 180, width: 980, height: 420 })
    await metrics('popover')
    await page.mouse.move(5, 5)
    // a glossary term tooltip in the detail card
    const term = page.locator('.detail .gl-term').first()
    if (await term.count()) {
      await term.scrollIntoViewIfNeeded(); await term.hover(); await page.waitForTimeout(400)
      const b = await term.boundingBox()
      await shotView('05-term-tip', ['light', 'dark'], { x: Math.max(0, b.x - 200), y: Math.max(0, b.y - 120), width: 620, height: 260 })
      await metrics('term-tip')
    }
    const icon = page.locator('.settings-line .gl-term-icon').first()
    await page.evaluate(() => scrollTo(0, 0))
    await icon.hover(); await page.waitForTimeout(400)
    await shotView('05-icon-tip', ['light', 'dark'], { x: 200, y: 60, width: 980, height: 300 })
    await page.mouse.move(5, 5)
  })

  await scene('playback', async () => {
    await page.evaluate(() => scrollTo(0, 0))
    for (let i = 0; i < 9; i++) await $('btn-step-back').click()
    await page.waitForTimeout(400)
    await shotView('06-mid-playback', ['light', 'dark'], { x: 200, y: 0, width: 980, height: 940 })
    await metrics('mid-playback')
    await $('btn-play').click({ timeout: 2000 }).catch(() => {})
    await page.waitForTimeout(700)
    await shotView('06-playing', ['light'], { x: 200, y: 200, width: 980, height: 400 })
    await $('btn-pause').click({ timeout: 2000 }).catch(() => {})
    await hoverAudit('top-playing', '.app')
  })

  await scene('focus', async () => {
    await page.evaluate(() => scrollTo(0, 0))
    await $('prompt-input').focus()
    const rows = []
    for (let i = 0; i < 45; i++) {
      await page.keyboard.press('Tab')
      const r = await page.evaluate(() => {
        const el = document.activeElement; if (!el || el === document.body) return null
        const s = getComputedStyle(el), p = el.parentElement ? getComputedStyle(el.parentElement) : null
        const cls = typeof el.className === 'string' ? el.className : ''
        return { el: el.tagName.toLowerCase() + '.' + cls.trim().replace(/\s+/g, '.') + (el.dataset.testid ? '#' + el.dataset.testid : ''),
          outline: s.outlineStyle === 'none' ? 'none' : `${s.outlineWidth} ${s.outlineStyle} ${s.outlineColor} +${s.outlineOffset}`,
          shadow: s.boxShadow === 'none' ? '-' : s.boxShadow.slice(0, 60), parentShadow: p && p.boxShadow !== 'none' ? p.boxShadow.slice(0, 60) : '-',
          visible: el.getBoundingClientRect().width > 0 }
      })
      if (r) rows.push(r)
      if (i === 3) await shotView('07-focus-a', ['light', 'dark'], { x: 200, y: 0, width: 980, height: 330 })
      if (i === 16) await shotView('07-focus-b', ['light'], { x: 200, y: 0, width: 980, height: 520 })
    }
    T.data.focus['top'] = rows; T.save()
  })

  await scene('glossary', async () => {
    await page.setViewportSize({ width: 1380, height: 2400 })
    await $('btn-help').first().click()
    await $('glossary-drawer').waitFor({ timeout: 5000 })
    await page.waitForTimeout(400)
    await shotEl('08-gl-index', $('glossary-drawer'), ['light', 'dark'], 0)
    await metrics('gl-index')
    await hoverAudit('gl-index', '.gl-drawer')
    const ids = await $('gl-index-row').evaluateAll((els) => els.map((e) => e.getAttribute('data-entry')))
    log('glossary entries', ids.join(','))
    for (const id of ids) {
      const row = page.locator(`[data-testid="gl-index-row"][data-entry="${id}"]`)
      if (!(await row.count())) await $('gl-all-entries').click({ timeout: 3000 }).catch(() => {})
      await row.click({ timeout: 3000 })
      await page.waitForTimeout(350)
      // the body may still be taller than the viewport: grow the viewport to fit
      const need = await page.evaluate(() => { const b = document.querySelector('.gl-drawer-body'); return b ? b.scrollHeight - b.clientHeight : 0 })
      if (need > 0) { await page.setViewportSize({ width: 1380, height: 2400 + need + 20 }); await page.waitForTimeout(200) }
      await shotEl(`08-gl-${id}`, $('glossary-drawer'), ['mlp', 'votes', 'softmax', 'how-to-read'].includes(id) ? ['light', 'dark'] : ['light'], 0)
      await metrics(`gl-${id}`)
      if (need > 0) await page.setViewportSize({ width: 1380, height: 2400 })
      await $('gl-all-entries').click({ timeout: 3000 }).catch(() => {})
      await page.waitForTimeout(150)
    }
    await $('gl-close').click()
    await page.setViewportSize({ width: 1380, height: 940 })
    // the drawer next to the page at normal height
    await $('btn-help').first().click(); await page.waitForTimeout(400)
    await shotView('08-gl-open-page', ['light'])
    await $('gl-close').click()
  })

  await scene('runs', async () => {
    await page.evaluate(() => scrollTo(0, 0))
    const prompts = ['The cat sat on the mat because it was tired', 'Once upon a time', 'Paris is the capital of', 'A very long prompt that keeps going and going to see how the run chip shortens its label text', 'Hello', 'The farmer bought a horse', '2 + 2 =']
    for (const p of prompts) {
      const before = await topNum()
      await $('prompt-input').fill(p)
      await $('btn-generate').click()
      await newerThan(before, 60000)
    }
    await $('btn-live').click({ timeout: 3000 }).catch(() => {})
    await page.waitForTimeout(400)
    const clip = { x: 200, y: 0, width: 980, height: 560 }
    await shotView('09-runs-rest', ['light', 'dark'], clip)
    await metrics('runs')
    await $('run-chip').nth(2).hover(); await page.waitForTimeout(300)
    await shotView('09-runs-hover', ['light'], clip)
    await $('run-chip').nth(2).getByTestId('btn-chip-pin').click(); await page.mouse.move(5, 5); await page.waitForTimeout(300)
    await shotView('09-runs-pinned', ['light'], clip)
    await hoverAudit('runs', '.run-shelf')
    await $('run-more').click(); await page.waitForTimeout(300)
    await shotView('09-runs-list', ['light', 'dark'], clip)
    await metrics('runs-list')
    await hoverAudit('runs-list', '.run-list')
    await page.keyboard.press('Escape')
    await $('run-chip').first().getByTestId('btn-chip-share').click({ timeout: 2000 }).catch(() => {})
    await page.waitForTimeout(300)
    await shotView('09-runs-share', ['light'], clip)
    await $('btn-compare-arm').click(); await page.waitForTimeout(300)
    await shotView('09-compare-armed', ['light', 'dark'], clip)
    await metrics('compare-armed')
    await $('run-chip-main').nth(1).click()
    await $('compare-view').waitFor({ timeout: 5000 })
    await $('cmp-tick').nth(2).click().catch(() => {})
    await page.waitForTimeout(500)
    await shotApp('10-compare')
    await metrics('compare')
    await hoverAudit('compare', '.compare-view')
    if (await $('dist-mode-combined').count()) { await $('dist-mode-combined').click(); await page.waitForTimeout(300); await shotEl('10-compare-combined', $('compare-view'), ['light']) ; await hoverAudit('compare-combined', '.compare-view') }
    await page.setViewportSize({ width: 400, height: 900 }); await page.waitForTimeout(300)
    await shotApp('10-compare-w400', ['light'])
    await metrics('compare-w400')
    await page.setViewportSize({ width: 1380, height: 940 })
    await $('btn-compare-exit').click()
  })

  await scene('errors', async () => {
    writeFileSync(`${OUT}/bad.json`, '{not json')
    await $('import-input').setInputFiles(`${OUT}/bad.json`)
    await $('import-error').waitFor({ timeout: 5000 })
    await shotView('11-import-error', ['light', 'dark'], { x: 200, y: 0, width: 980, height: 420 })
    await metrics('import-error')
  })

  await scene('narrow', async () => {
    for (const w of [1024, 768, 400]) {
      await page.setViewportSize({ width: w, height: 900 }); await page.waitForTimeout(300)
      for (const [i, nm] of [[0, 'tokenizer'], [2, 'layers'], [4, 'softmax']]) {
        await $('stage-card').nth(i).click(); await page.waitForTimeout(500)
        if (w === 400 || nm === 'layers') await shotApp(`12-w${w}-${nm}`, w === 400 && nm === 'layers' ? ['light', 'dark'] : ['light'])
        await metrics(`w${w}-${nm}`)
      }
    }
    await page.setViewportSize({ width: 1380, height: 940 })
  })
  log('DONE')
} catch (e) { console.error(e) } finally { await context.close(); stop(); process.exit(0) }

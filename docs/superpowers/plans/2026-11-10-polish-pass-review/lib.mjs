// shared helpers for the UI audit scripts
import { spawn } from 'node:child_process'
import { mkdirSync, writeFileSync, appendFileSync } from 'node:fs'

export const REPO = '/Users/markus/IdeaProjects/llm-visualizer'
export const PORT = 5199
export const URL = `http://localhost:${PORT}/`

export const startServer = async () => {
  const up = async () => { try { return (await fetch(URL)).ok } catch { return false } }
  if (await up()) return () => {}
  const server = spawn('npm', ['run', 'dev', '--', '--port', String(PORT)], { stdio: 'ignore', detached: true, cwd: REPO })
  const stop = () => { try { process.kill(-server.pid, 'SIGTERM') } catch {} }
  process.on('exit', stop)
  for (let i = 0; i < 60; i++) { if (await up()) return stop; await new Promise((r) => setTimeout(r, 500)) }
  throw new Error('no server')
}

// functions installed on window.__a in the page
export const installPageFns = (page) => page.addInitScript(() => {
  const parse = (c) => {
    if (!c) return null
    let m = c.match(/^rgba?\(([^)]+)\)/)
    if (m) { const p = m[1].split(/[,\s/]+/).filter(Boolean).map(Number); return { r: p[0], g: p[1], b: p[2], a: p[3] ?? 1 } }
    m = c.match(/^color\(srgb ([^)]+)\)/)
    if (m) { const p = m[1].split(/[\s/]+/).filter(Boolean).map(Number); return { r: p[0] * 255, g: p[1] * 255, b: p[2] * 255, a: p[3] ?? 1 } }
    return null
  }
  const over = (top, base) => ({ r: top.r * top.a + base.r * (1 - top.a), g: top.g * top.a + base.g * (1 - top.a), b: top.b * top.a + base.b * (1 - top.a), a: 1 })
  const effBg = (el) => {
    const chain = []
    for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const c = parse(getComputedStyle(e).backgroundColor)
      if (c && c.a > 0) { chain.push(c); if (c.a >= .99) break }
    }
    let base = { r: 255, g: 255, b: 255, a: 1 }
    for (let i = chain.length - 1; i >= 0; i--) base = over(chain[i], base)
    return base
  }
  const lum = (c) => { const f = (v) => { v /= 255; return v <= .03928 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4 }; return .2126 * f(c.r) + .7152 * f(c.g) + .0722 * f(c.b) }
  const contrast = (a, b) => { const la = lum(a), lb = lum(b); return (Math.max(la, lb) + .05) / (Math.min(la, lb) + .05) }
  const hex = (c) => c ? '#' + [c.r, c.g, c.b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('') + (c.a < .99 ? `@${c.a.toFixed(2)}` : '') : '-'
  const visible = (el) => {
    const r = el.getBoundingClientRect()
    if (r.width < 1 || r.height < 1) return false
    const s = getComputedStyle(el)
    return s.visibility !== 'hidden' && s.display !== 'none' && Number(s.opacity) > 0
  }
  const STATE = ['aria-pressed', 'data-active', 'data-selected', 'aria-current', 'aria-expanded', 'data-role', 'data-top', 'data-chosen', 'data-pinned', 'data-lit', 'data-fork']
  const keyOf = (el) => {
    const cls = typeof el.className === 'string' ? el.className.trim().split(/\s+/).filter(Boolean).join('.') : (el.className?.baseVal ?? '')
    let k = el.tagName.toLowerCase() + (cls ? '.' + cls : '')
    for (const a of STATE) if (el.hasAttribute(a) && el.getAttribute(a) !== 'false') k += `[${a}=${el.getAttribute(a)}]`
    if (el.disabled) k += '[disabled]'
    if (!cls && el.dataset?.testid) k += `#${el.dataset.testid}`
    return k
  }
  const INTERACTIVE = 'button, a[href], input, select, textarea, summary, [role="button"], [tabindex]:not([tabindex="-1"])'
  const px = (v) => Math.round(parseFloat(v) * 100) / 100
  const sig = (el) => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect()
    const bg = parse(s.backgroundColor)
    const o = {
      h: Math.round(r.height * 10) / 10, fs: px(s.fontSize), fw: s.fontWeight, ff: s.fontFamily.split(',')[0].replace(/["']/g, ''),
      rad: s.borderTopLeftRadius === s.borderBottomRightRadius ? s.borderTopLeftRadius : s.borderRadius,
      pad: `${px(s.paddingTop)} ${px(s.paddingRight)} ${px(s.paddingBottom)} ${px(s.paddingLeft)}`,
      bg: bg && bg.a > 0 ? hex(bg) : '-', color: hex(parse(s.color)),
      bw: px(s.borderTopWidth), bc: px(s.borderTopWidth) > 0 ? hex(parse(s.borderTopColor)) : '-', bs: px(s.borderTopWidth) > 0 ? s.borderTopStyle : '-',
      shadow: s.boxShadow === 'none' ? '-' : 'y', trans: s.transitionDuration === '0s' ? '-' : s.transitionProperty + ' ' + s.transitionDuration,
      cursor: s.cursor,
    }
    return o
  }
  // every visible interactive element, and every element that looks like a box (radius or border or fill)
  const inventory = () => {
    const out = new Map()
    for (const el of document.body.querySelectorAll('*')) {
      if (el.closest('svg') && el.tagName.toLowerCase() !== 'svg') continue
      if (!visible(el)) continue
      const s = getComputedStyle(el)
      const inter = el.matches(INTERACTIVE)
      const boxy = parseFloat(s.borderTopLeftRadius) > 0 || parseFloat(s.borderTopWidth) > 0 || (parse(s.backgroundColor)?.a ?? 0) > 0
      if (!inter && !boxy) continue
      const k = keyOf(el), g = sig(el)
      const id = k + ' | ' + JSON.stringify(g)
      const cur = out.get(id)
      if (cur) cur.n++
      else out.set(id, { key: k, inter, n: 1, text: (el.textContent ?? '').trim().slice(0, 24), ...g })
    }
    return [...out.values()]
  }
  // every distinct text style in use
  const textStyles = () => {
    const out = new Map()
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (!n.textContent.trim()) continue
      const el = n.parentElement
      if (!el || !visible(el)) continue
      const s = getComputedStyle(el)
      const svg = !!el.closest('svg')
      const col = parse(svg ? s.fill : s.color)
      const bg = effBg(el)
      const id = [px(s.fontSize), s.fontWeight, s.fontFamily.split(',')[0].replace(/["']/g, ''), s.fontStyle, s.textTransform, s.letterSpacing, hex(col)].join(' | ')
      const cr = col ? Math.round(contrast(over({ ...col, a: (col.a ?? 1) * Number(s.opacity) }, bg), bg) * 100) / 100 : null
      const cur = out.get(id)
      if (cur) { cur.n++; if (cr !== null && cr < cur.minCr) { cur.minCr = cr; cur.minEl = keyOf(el); cur.minText = n.textContent.trim().slice(0, 30) } if (!cur.keys.includes(keyOf(el)) && cur.keys.length < 6) cur.keys.push(keyOf(el)) }
      else out.set(id, { id, n: 1, minCr: cr ?? 99, minEl: keyOf(el), minText: n.textContent.trim().slice(0, 30), keys: [keyOf(el)] })
    }
    return [...out.values()]
  }
  const overflow = () => {
    const de = document.documentElement
    const res = { docScroll: de.scrollWidth - innerWidth, viewport: innerWidth, past: [], truncated: [], scrollers: [], wrapped: [] }
    const app = document.querySelector('.app')?.getBoundingClientRect()
    for (const el of document.body.querySelectorAll('*')) {
      if (!visible(el)) continue
      const s = getComputedStyle(el), r = el.getBoundingClientRect()
      if (el.scrollWidth > el.clientWidth + 1 && s.overflowX !== 'visible') {
        if (s.textOverflow === 'ellipsis') res.truncated.push(keyOf(el) + ' "' + (el.textContent ?? '').trim().slice(0, 40) + '"')
        else if (s.overflowX === 'auto' || s.overflowX === 'scroll') res.scrollers.push(`${keyOf(el)} ${el.clientWidth}/${el.scrollWidth}`)
      }
      if (app && !el.closest('.gl-drawer') && s.position !== 'fixed' && r.right > Math.min(innerWidth, app.right) + 1.5) {
        let clipped = false
        for (let p = el.parentElement; p; p = p.parentElement) { const o = getComputedStyle(p).overflowX; if (o !== 'visible') { clipped = true; break } }
        if (!clipped) res.past.push(`${keyOf(el)} right=${Math.round(r.right)} (limit ${Math.round(Math.min(innerWidth, app.right))})`)
      }
    }
    res.past = [...new Set(res.past)].slice(0, 25); res.truncated = [...new Set(res.truncated)].slice(0, 25); res.scrollers = [...new Set(res.scrollers)]
    return res
  }
  const blocks = () => {
    const sels = ['.app-header', '.prompt-box', '.model-slot', '.link-note', '.run-shelf', '.token-stream', '.pipeline-band', '.controls', '.detail', '.compare-view', '.mlp-card', '.gl-section', '.layers-nav']
    const out = []
    for (const sel of sels) for (const el of document.querySelectorAll(sel)) {
      if (!visible(el)) continue
      const r = el.getBoundingClientRect(), s = getComputedStyle(el)
      out.push({ sel, x: Math.round(r.left), y: Math.round(r.top + scrollY), w: Math.round(r.width), h: Math.round(r.height), pad: `${px(s.paddingTop)} ${px(s.paddingRight)} ${px(s.paddingBottom)} ${px(s.paddingLeft)}`, mt: px(s.marginTop), mb: px(s.marginBottom) })
    }
    return out.sort((a, b) => a.y - b.y)
  }
  // hover audit: tag one representative per key, snapshot rects of all interactive elements
  const tagReps = () => {
    const seen = new Set(); let i = 0; const reps = []
    for (const el of document.body.querySelectorAll(INTERACTIVE)) {
      if (!visible(el)) continue
      const k = keyOf(el)
      if (seen.has(k)) continue
      seen.add(k); el.setAttribute('data-audit', String(i)); reps.push({ i, key: k, text: (el.textContent ?? el.getAttribute('aria-label') ?? '').trim().slice(0, 24) }); i++
    }
    return reps
  }
  const snap = () => {
    const m = []
    for (const el of document.body.querySelectorAll(INTERACTIVE + ', .token, .token-chip, .run-chip, .stage-card')) {
      if (!visible(el)) continue
      const r = el.getBoundingClientRect(); m.push([el, r.left, r.top, r.width, r.height])
    }
    window.__snap = m
  }
  const state = (i) => {
    const el = document.querySelector(`[data-audit="${i}"]`)
    if (!el) return null
    const s = getComputedStyle(el)
    const bg = effBg(el), col = parse(s.color)
    let moved = null, max = 0
    for (const [o, l, t, w, h] of window.__snap ?? []) {
      if (o === el || el.contains(o) || o.contains(el) || !o.isConnected) continue
      const r = o.getBoundingClientRect()
      const d = Math.max(Math.abs(r.left - l), Math.abs(r.top - t), Math.abs(r.width - w), Math.abs(r.height - h))
      if (d > max) { max = d; moved = keyOf(o) }
    }
    return { bg: hex(bg), rawBg: s.backgroundColor, color: hex(col), bc: px(s.borderTopWidth) > 0 ? hex(parse(s.borderTopColor)) : '-', cr: col ? Math.round(contrast(col, bg) * 100) / 100 : null,
      shadow: s.boxShadow === 'none' ? '-' : 'y', filter: s.filter, opacity: s.opacity, textDec: s.textDecorationLine, shift: Math.round(max * 10) / 10, moved, outline: s.outlineStyle !== 'none' ? `${s.outlineWidth} ${s.outlineStyle} ${hex(parse(s.outlineColor))} +${s.outlineOffset}` : '-' }
  }
  window.__a = { inventory, textStyles, overflow, blocks, tagReps, snap, state }
})

export const makeTools = (page, OUT) => {
  mkdirSync(OUT, { recursive: true })
  const $ = (id) => page.getByTestId(id)
  const log = (...a) => { console.log(...a); appendFileSync(`${OUT}/log.txt`, a.join(' ') + '\n') }
  let theme = 'light'
  const setTheme = async (t) => { theme = t; await page.emulateMedia({ colorScheme: t }); await page.waitForTimeout(250) }
  const data = { inventory: {}, text: {}, overflow: {}, blocks: {}, hover: {}, focus: {} }
  const save = () => writeFileSync(`${OUT}/data.json`, JSON.stringify(data, null, 1))
  const metrics = async (label) => {
    try {
      data.inventory[label] = await page.evaluate(() => window.__a.inventory())
      data.text[label] = await page.evaluate(() => window.__a.textStyles())
      data.overflow[label] = await page.evaluate(() => window.__a.overflow())
      data.blocks[label] = await page.evaluate(() => window.__a.blocks())
      save()
    } catch (e) { log('METRICS FAILED', label, String(e).slice(0, 200)) }
  }
  // hover every distinct interactive element once; record the hovered style and what else moved
  const hoverAudit = async (label, scope = 'body') => {
    try {
      const reps = await page.evaluate(() => window.__a.tagReps())
      const rows = []
      for (const rep of reps) {
        const loc = page.locator(`${scope} [data-audit="${rep.i}"]`)
        if (!(await loc.count())) continue
        try {
          await page.mouse.move(2, 2)
          await page.waitForTimeout(30)
          const rest = await page.evaluate((i) => { window.__a.snap(); return window.__a.state(i) }, rep.i)
          await loc.scrollIntoViewIfNeeded({ timeout: 800 })
          await page.mouse.move(2, 2)
          await page.evaluate(() => window.__a.snap())
          await loc.hover({ timeout: 800 })
          await page.waitForTimeout(220)
          const hov = await page.evaluate((i) => window.__a.state(i), rep.i)
          rows.push({ ...rep, rest, hov })
        } catch (e) { rows.push({ ...rep, err: String(e).split('\n')[0].slice(0, 80) }) }
      }
      await page.mouse.move(2, 2)
      await page.evaluate(() => document.querySelectorAll('[data-audit]').forEach((e) => e.removeAttribute('data-audit')))
      data.hover[label] = rows; save()
      log('hover audit', label, rows.length)
    } catch (e) { log('HOVER FAILED', label, String(e).slice(0, 200)) }
  }
  const themes = ['light', 'dark']
  // slices a tall region into files no taller than 1250px
  const shotRegion = async (name, box, which = themes) => {
    for (const th of which) {
      await setTheme(th)
      mkdirSync(`${OUT}/${th}`, { recursive: true })
      const n = Math.max(1, Math.ceil(box.height / 1250))
      for (let k = 0; k < n; k++) {
        const path = `${OUT}/${th}/${name}${n > 1 ? `-${k + 1}` : ''}.png`
        const h = Math.min(1250, box.height - k * 1250)
        try { await page.screenshot({ path, fullPage: true, clip: { x: Math.max(0, box.x), y: box.y + k * 1250, width: box.width, height: h }, timeout: 15000 }) }
        catch (e) { log('SHOT FAILED', name, th, String(e).slice(0, 120)) }
      }
    }
    await setTheme('light')
    log('shot', name)
  }
  const absBox = async (loc, pad = 0) => {
    const b = await loc.boundingBox({ timeout: 5000 })
    const sy = await page.evaluate(() => scrollY)
    return { x: b.x - pad, y: b.y + sy - pad, width: b.width + 2 * pad, height: b.height + 2 * pad }
  }
  const shotEl = async (name, loc, which = themes, pad = 8) => {
    try { await shotRegion(name, await absBox(loc, pad), which) } catch (e) { log('SHOT FAILED', name, String(e).slice(0, 160)) }
  }
  // the app column from y0 to y1 (page coordinates); y1 defaults to the page bottom
  const shotApp = async (name, which = themes, y0 = 0, y1 = null) => {
    try {
      const app = await page.locator('.app').boundingBox()
      const sy = await page.evaluate(() => scrollY)
      const bottom = y1 ?? (app.y + sy + app.height)
      await shotRegion(name, { x: Math.max(0, app.x - 10), y: y0, width: Math.min(app.width + 20, (await page.viewportSize()).width), height: bottom - y0 }, which)
    } catch (e) { log('SHOT FAILED', name, String(e).slice(0, 160)) }
  }
  const shotView = async (name, which = themes, clip = undefined) => {
    for (const th of which) {
      await setTheme(th); mkdirSync(`${OUT}/${th}`, { recursive: true })
      try { await page.screenshot({ path: `${OUT}/${th}/${name}.png`, clip, timeout: 15000 }) } catch (e) { log('SHOT FAILED', name, th, String(e).slice(0, 120)) }
    }
    await setTheme('light'); log('shot', name)
  }
  const scene = async (name, fn) => { try { await fn(); log('scene ok', name) } catch (e) { log('SCENE FAILED', name, String(e).slice(0, 400)) } }
  const topNum = async () => {
    const t = await $('run-chip').first().textContent({ timeout: 1000 }).catch(() => null)
    return t ? Number(t.match(/#(\d+)/)?.[1]) : 0
  }
  const newerThan = (n, timeout = 180000) => page.waitForFunction((n) => {
    const t = document.querySelector('[data-testid="run-chip"]')?.textContent
    return t && Number(t.match(/#(\d+)/)[1]) > n
  }, n, { timeout })
  const openExamples = async () => { if (!(await $('example-chip').count())) await $('toggle-examples').click() }
  const unstick = () => page.addStyleTag({ content: '.controls, .layers-nav { position: static !important; }' })
  return { $, log, setTheme, metrics, hoverAudit, shotRegion, shotEl, shotApp, shotView, scene, topNum, newerThan, openExamples, unstick, data, save, absBox }
}

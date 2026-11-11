import type { Page } from '@playwright/test'

export type Control = { index: number; key: string; text: string }
export type ControlState = {
  background: string // the control's own computed background, normalised (hex, or hex@alpha)
  border: string // the control's top border colour, normalised; 'transparent' when it has no border
  text: string // text colour, normalised
  contrast: number | null // text colour over the effective background behind the control
  disabled: boolean
  noText: boolean // a slider, checkbox or radio button: it has no text, so there is no contrast to read
}

/** The page-side API, defined by installProbe as window.__polish. */
export type Polish = {
  controls(): Control[]
  state(index: number): ControlState | null
  snapshot(): void
  movedVertically(exceptIndex: number): string[]
  ring(index: number): { signature: string; visible: boolean } | null
  /** Records every control's resting ring, so that `activeRing` can tell a ring that appeared from a look that was already there. */
  rememberRings(): number
  activeRing(): { key: string; text: string; signature: string; visible: boolean; changed: boolean | null } | null
  unnamedControls(): { checked: number; unnamed: string[] }
  textIssues(): string[]
  clippedLabels(): string[]
  overflowingTips(): string[]
  sidewaysScroll(): number
  palette(): Record<string, string>
}

declare global {
  interface Window {
    __polish: Polish
  }
}

/** Defines window.__polish before any page script runs. The function body runs in the page, so it cannot use anything from this module. */
export async function installProbe(page: Page): Promise<void> {
  await page.addInitScript(() => {
    type RGBA = { r: number; g: number; b: number; a: number }
    const parse = (c: string | null): RGBA | null => {
      if (!c) return null
      let m = c.match(/^rgba?\(([^)]+)\)/)
      if (m) {
        const p = m[1].split(/[,\s/]+/).filter(Boolean).map(Number)
        return { r: p[0], g: p[1], b: p[2], a: p[3] ?? 1 }
      }
      m = c.match(/^color\(srgb ([^)]+)\)/)
      if (m) {
        const p = m[1].split(/[\s/]+/).filter(Boolean).map(Number)
        return { r: p[0] * 255, g: p[1] * 255, b: p[2] * 255, a: p[3] ?? 1 }
      }
      return null
    }
    const over = (top: RGBA, base: RGBA): RGBA => ({
      r: top.r * top.a + base.r * (1 - top.a),
      g: top.g * top.a + base.g * (1 - top.a),
      b: top.b * top.a + base.b * (1 - top.a),
      a: 1,
    })
    const hex = (c: RGBA | null): string =>
      c ? '#' + [c.r, c.g, c.b].map((v) => Math.round(v).toString(16).padStart(2, '0')).join('') + (c.a < 0.99 ? `@${c.a.toFixed(2)}` : '') : '-'
    const norm = (css: string): string => {
      const c = parse(css)
      if (!c) {
        if (css === 'transparent') return 'transparent'
        throw new Error(`cannot parse the colour "${css}"`)
      }
      return c.a === 0 ? 'transparent' : hex(c)
    }
    const effectiveBackground = (el: Element): RGBA => {
      const chain: RGBA[] = []
      for (let e: Element | null = el; e && e.nodeType === 1; e = e.parentElement) {
        const c = parse(getComputedStyle(e).backgroundColor)
        if (c && c.a > 0) {
          chain.push(c)
          if (c.a >= 0.99) break
        }
      }
      let base: RGBA = { r: 255, g: 255, b: 255, a: 1 }
      for (let i = chain.length - 1; i >= 0; i--) base = over(chain[i], base)
      return base
    }
    const luminance = (c: RGBA) => {
      const f = (v: number) => {
        v /= 255
        return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
      }
      return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b)
    }
    const contrastOf = (a: RGBA, b: RGBA) => {
      const la = luminance(a)
      const lb = luminance(b)
      return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
    }
    const visible = (el: Element): boolean => {
      const r = el.getBoundingClientRect()
      if (r.width < 1 || r.height < 1) return false
      const s = getComputedStyle(el)
      return s.visibility !== 'hidden' && s.display !== 'none' && Number(s.opacity) > 0
    }
    const STATE_ATTRS = ['aria-pressed', 'data-active', 'data-selected', 'aria-current', 'aria-expanded', 'data-role', 'data-top', 'data-chosen', 'data-pinned', 'data-lit', 'data-fork']
    const keyOf = (el: Element): string => {
      const raw = el.getAttribute('class') ?? ''
      const cls = raw.trim().split(/\s+/).filter(Boolean).join('.')
      let k = el.tagName.toLowerCase() + (cls ? '.' + cls : '')
      for (const a of STATE_ATTRS) if (el.hasAttribute(a) && el.getAttribute(a) !== 'false') k += `[${a}=${el.getAttribute(a)}]`
      // a control inside a stateful wrapper (the button in an active run chip) is a different control
      const parent = el.parentElement
      if (parent) for (const a of STATE_ATTRS) if (parent.hasAttribute(a) && parent.getAttribute(a) !== 'false') k += `<${a}=${parent.getAttribute(a)}>`
      if ((el as HTMLButtonElement).disabled) k += '[disabled]'
      if (!cls && (el as HTMLElement).dataset?.testid) k += `#${(el as HTMLElement).dataset.testid}`
      return k
    }
    const INTERACTIVE = 'button, a[href], input, select, textarea, summary, [role="button"], [tabindex]:not([tabindex="-1"])'
    const MOVERS = INTERACTIVE + ', .token, .tok, .run-chip, .stage-card'

    // A name is an aria-label, an aria-labelledby that resolves to text, an associated label, or own text with a letter or digit.
    // Symbols alone ("×", "▶") and a title alone are not names.
    const WORDY = /[\p{L}\p{N}]/u
    const nameOf = (el: Element): string => {
      const label = el.getAttribute('aria-label')?.trim()
      if (label) return label
      const by = el.getAttribute('aria-labelledby')
      if (by) {
        const t = by.split(/\s+/).map((id) => document.getElementById(id)?.textContent ?? '').join(' ').trim()
        if (WORDY.test(t)) return t
      }
      const labels = (el as HTMLInputElement).labels
      if (labels && [...labels].some((l) => WORDY.test(l.textContent ?? ''))) return 'label'
      if (el.matches('input, select, textarea')) return ''
      const text = (el.textContent ?? '').trim()
      if (WORDY.test(text)) return text
      const img = el.querySelector('img[alt]:not([alt=""])')
      return img ? img.getAttribute('alt')! : ''
    }
    const ringOfOne = (el: Element) => {
      const s = getComputedStyle(el)
      const outlineVisible = s.outlineStyle !== 'none' && parseFloat(s.outlineWidth) > 0
      const shadowVisible = s.boxShadow !== 'none'
      return {
        signature: `${s.outlineStyle} ${s.outlineWidth} ${s.outlineColor} | ${s.boxShadow}`,
        visible: outlineVisible || shadowVisible,
      }
    }
    // a field may show its ring on its container (the prompt line, the settings chip): read the element and its two nearest ancestors
    const ringOf = (el: Element) => {
      const tag = el.tagName
      if (tag !== 'INPUT' && tag !== 'SELECT' && tag !== 'TEXTAREA') return ringOfOne(el)
      const three = [el, el.parentElement, el.parentElement?.parentElement].filter((e): e is Element => !!e).map(ringOfOne)
      return { signature: three.map((r) => r.signature).join(' || '), visible: three.some((r) => r.visible) }
    }

    let snapshotRects: [Element, number, number][] = []
    let restingRings = new WeakMap<Element, string>()

    const polish = {
      controls() {
        document.querySelectorAll('[data-polish]').forEach((e) => e.removeAttribute('data-polish'))
        const seen = new Set<string>()
        const out: { index: number; key: string; text: string }[] = []
        for (const el of document.body.querySelectorAll(INTERACTIVE)) {
          if (!visible(el)) continue
          const k = keyOf(el)
          if (seen.has(k)) continue
          seen.add(k)
          el.setAttribute('data-polish', String(out.length))
          out.push({ index: out.length, key: k, text: (el.textContent || el.getAttribute('aria-label') || '').trim().slice(0, 24) })
        }
        return out
      },
      state(index: number) {
        const el = document.querySelector(`[data-polish="${index}"]`)
        if (!el) return null
        const s = getComputedStyle(el)
        const text = parse(s.color)
        const bg = effectiveBackground(el)
        // a slider, checkbox or radio button, or a bar-shaped button with neither text, nor an icon, nor text drawn by
        // a pseudo-element: nothing is written on it
        const pseudo = (which: string) => { const c = getComputedStyle(el, which).content; return c !== 'none' && c !== 'normal' && c !== '""' }
        const noText = (el instanceof HTMLInputElement && ['range', 'checkbox', 'radio'].includes(el.type))
          || (el instanceof HTMLButtonElement && !(el.textContent ?? '').trim() && !el.querySelector('svg') && !pseudo('::before') && !pseudo('::after'))
        return {
          background: norm(s.backgroundColor),
          border: parseFloat(s.borderTopWidth) > 0 && s.borderTopStyle !== 'none' ? norm(s.borderTopColor) : 'transparent',
          text: hex(text),
          contrast: text && !noText ? Math.round(contrastOf(over({ ...text, a: text.a * Number(s.opacity) }, bg), bg) * 100) / 100 : null,
          disabled: (el as HTMLButtonElement).disabled === true || el.getAttribute('aria-disabled') === 'true',
          noText,
        }
      },
      rememberRings() {
        restingRings = new WeakMap()
        let count = 0
        for (const el of document.body.querySelectorAll(INTERACTIVE)) {
          restingRings.set(el, ringOf(el).signature)
          count++
        }
        return count
      },
      activeRing() {
        const el = document.activeElement
        if (!el || el === document.body || el === document.documentElement) return null
        const ring = ringOf(el)
        const resting = restingRings.get(el)
        return {
          key: keyOf(el), text: (el.textContent || el.getAttribute('aria-label') || '').trim().slice(0, 24), ...ring,
          changed: resting === undefined ? null : ring.signature !== resting, // null: the control was not there when the rings were recorded
        }
      },
      ring(index: number) {
        const el = document.querySelector(`[data-polish="${index}"]`)
        return el ? ringOf(el) : null
      },
      unnamedControls() {
        const out = new Set<string>()
        let checked = 0
        for (const el of document.body.querySelectorAll(INTERACTIVE)) {
          if (!visible(el)) continue
          checked++
          if (!nameOf(el)) out.add(`${keyOf(el)} "${(el.textContent ?? '').trim().slice(0, 20)}"`)
        }
        return { checked, unnamed: [...out] }
      },
      snapshot() {
        snapshotRects = []
        for (const el of document.body.querySelectorAll(MOVERS)) {
          if (!visible(el)) continue
          const r = el.getBoundingClientRect()
          snapshotRects.push([el, r.top + scrollY, r.height])
        }
      },
      movedVertically(exceptIndex: number) {
        const self = document.querySelector(`[data-polish="${exceptIndex}"]`)
        const moved: string[] = []
        for (const [el, top, height] of snapshotRects) {
          if (!el.isConnected) continue
          if (self && self.contains(el)) continue // the control itself and what it holds; its ancestors changing height is a move too
          const r = el.getBoundingClientRect()
          const dTop = r.top + scrollY - top
          const dHeight = r.height - height
          if (Math.abs(dTop) > 0.5 || Math.abs(dHeight) > 0.5) moved.push(`${keyOf(el)} top ${dTop.toFixed(1)} height ${dHeight.toFixed(1)}`)
        }
        return moved
      },
      textIssues() {
        const rules: [string, RegExp][] = [
          ['hyphen as minus', /(^|[\s(~≈+:])-\d/],
          ['thousands comma', /\d,\d{3}(?!\d)/],
          ['space before %', /\d %/],
          ['tilde before sign or digit', /~[-+−\d]/],
          // groups of three are separated by a narrow no-break space (U+202F), and only from 10 000 up
          ['digits grouped with an ordinary or no-break space', /(?<![\d.,])\d{1,3}[ \u00a0]\d{3}(?!\d)/],
          ['four-digit number grouped', /(?<![\d.,])\d[\s\u00a0\u202f]\d{3}(?![\d.,]?\d)/],
        ]
        const issues = new Set<string>()
        // React renders "~", the sign and the number as neighbouring text nodes, so test the joined direct text of each element
        for (const el of document.body.querySelectorAll('*')) {
          let text = ''
          for (const child of el.childNodes) if (child.nodeType === Node.TEXT_NODE) text += child.textContent ?? ''
          if (!text.trim() || !visible(el)) continue
          if (el.closest('textarea, input, .token, .tok, code, script, style')) continue
          for (const [name, re] of rules) {
            const m = text.match(re)
            if (m) issues.add(`${name}: "${text.trim().slice(Math.max(0, (m.index ?? 0) - 12), (m.index ?? 0) + 18)}" in ${keyOf(el)}`)
          }
        }
        return [...issues]
      },
      clippedLabels() {
        const issues = new Set<string>()
        for (const t of document.body.querySelectorAll('svg text')) {
          const svg = (t as SVGTextElement).ownerSVGElement
          if (!svg || !visible(svg)) continue
          const b = t.getBoundingClientRect()
          const s = svg.getBoundingClientRect()
          if (b.width < 1 && b.height < 1) continue
          if (b.left < s.left - 1 || b.right > s.right + 1 || b.top < s.top - 1 || b.bottom > s.bottom + 1)
            issues.add(`svg text "${(t.textContent ?? '').trim().slice(0, 30)}" leaves its svg`)
        }
        const AXIS = '[class*="axis"], [class*="-tick"], .bar-label, .attn-label, .embed-col-label, .mlp-chart-label, .knockout-label, .gl-plot-label'
        for (const el of document.body.querySelectorAll(AXIS)) {
          if (el.closest('svg') || !visible(el)) continue
          const h = el as HTMLElement
          if (h.scrollWidth > h.clientWidth + 1 && getComputedStyle(h).textOverflow !== 'ellipsis')
            issues.add(`${keyOf(el)} "${(el.textContent ?? '').trim().slice(0, 30)}" ${h.clientWidth}/${h.scrollWidth}`)
        }
        return [...issues]
      },
      overflowingTips() {
        const issues: string[] = []
        for (const tip of document.querySelectorAll('[data-testid="gl-tip"]')) {
          if (!visible(tip)) continue
          const t = tip as HTMLElement
          const tr = t.getBoundingClientRect()
          const text = (t.textContent ?? '').trim().slice(0, 30)
          if (t.scrollWidth > t.clientWidth + 1 || t.scrollHeight > t.clientHeight + 1) issues.push(`tip "${text}" scroll ${t.scrollWidth}x${t.scrollHeight} in ${t.clientWidth}x${t.clientHeight}`)
          for (const child of t.querySelectorAll('*')) {
            const r = child.getBoundingClientRect()
            if (r.width > 0 && (r.right > tr.right + 1 || r.left < tr.left - 1 || r.bottom > tr.bottom + 1 || r.top < tr.top - 1))
              issues.push(`tip "${text}" child ${child.tagName.toLowerCase()} leaves the box`)
          }
          // a Range over the text catches overflow that scroll sizes miss
          const range = document.createRange()
          range.selectNodeContents(t)
          for (const r of range.getClientRects()) {
            if (r.right > tr.right + 1 || r.left < tr.left - 1 || r.bottom > tr.bottom + 1 || r.top < tr.top - 1) {
              issues.push(`tip "${text}" text leaves the box`)
              break
            }
          }
        }
        return [...new Set(issues)]
      },
      sidewaysScroll() {
        return Math.max(0, document.documentElement.scrollWidth - innerWidth)
      },
      palette() {
        const out: Record<string, string> = {}
        for (const name of ['card', 'washi', 'white', 'seg-ground', 'fill', 'fill-hover', 'ai', 'shu', 'shu-deep']) {
          const probe = document.createElement('div')
          probe.style.cssText = `position:absolute;visibility:hidden;background-color:var(--${name})`
          document.body.appendChild(probe)
          out[name] = norm(getComputedStyle(probe).backgroundColor)
          probe.remove()
        }
        return out
      },
    }
    // The init script is stringified, so expose a plain object on window.
    ;(window as unknown as { __polish: typeof polish }).__polish = polish
  })
}

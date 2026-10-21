import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { getEntry } from './registry'
import { consumeReturnedFocus, useGlossaryStore } from './store'
import type { GlossaryId } from './types'

const CLASS = { text: 'gl-term', chip: 'embed-callout gl-term gl-term-chip', icon: 'gl-term-icon' } as const
// .gl-tip is at most 280px wide plus padding; keep it inside the viewport
const TIP_SPACE = 296
const GAP = 4
const EDGE = 8

// A word in a caption (or a callout chip, or a heading's ⓘ) that shows the
// entry's summary (or its own tip) on hover or focus and opens the entry on click.
export function Term({ id, section, variant = 'text', label, testId, tip: tipText, children }: {
  id: GlossaryId
  section?: string
  variant?: 'text' | 'chip' | 'icon'
  label?: string
  testId?: string
  // replaces the summary in the tooltip, e.g. a chip naming the section it opens
  tip?: string
  children: ReactNode
}) {
  const openEntry = useGlossaryStore((s) => s.openEntry)
  // viewport position of the tooltip, or null when hidden; fixed positioning
  // keeps it out of the drawer's scroll clipping. above is the term's top edge,
  // used when the tooltip would run past the bottom of the window.
  const [tip, setTip] = useState<{ left: number; top: number; above: number } | null>(null)
  const tipRef = useRef<HTMLSpanElement>(null)
  const tipId = useId()
  // The height is known only once the tooltip is rendered: before the browser
  // paints, move it above the term if it does not fit below.
  useLayoutEffect(() => {
    const el = tipRef.current
    if (!tip || !el) return
    const h = el.offsetHeight
    const flipped = tip.above - GAP - h
    if (tip.top + h > window.innerHeight - EDGE && flipped >= EDGE && flipped !== tip.top) setTip({ ...tip, top: flipped })
  }, [tip])
  // Escape hides an open tooltip (WCAG 1.4.13)
  useEffect(() => {
    if (!tip) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setTip(null) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [tip])
  const entry = getEntry(id)
  if (!entry) return <span>{children}</span>
  return (
    <span className="gl-term-wrap">
      <button type="button" className={CLASS[variant]} data-testid={testId ?? 'gl-term'}
        data-term={id} data-section={section} aria-label={label}
        aria-describedby={tip ? tipId : undefined}
        onMouseEnter={(e) => setTip(place(e.currentTarget))} onMouseLeave={() => setTip(null)}
        onFocus={(e) => { if (!consumeReturnedFocus(e.currentTarget)) setTip(place(e.currentTarget)) }}
        onBlur={() => setTip(null)}
        onClick={(e) => { setTip(null); openEntry(entry.id, section, e.currentTarget) }}>
        {children}
      </button>
      {tip && (
        <span ref={tipRef} role="tooltip" id={tipId} data-testid="gl-tip" className="gl-tip" style={{ left: tip.left, top: tip.top }}>
          {tipText ?? entry.summary} <em>Click to read more.</em>
        </span>
      )}
    </span>
  )
}

function place(el: HTMLElement): { left: number; top: number; above: number } {
  const r = el.getBoundingClientRect()
  return { left: Math.max(EDGE, Math.min(r.left, window.innerWidth - TIP_SPACE)), top: r.bottom + GAP, above: r.top }
}

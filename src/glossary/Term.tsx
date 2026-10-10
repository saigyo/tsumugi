import { useId, useState, type ReactNode } from 'react'
import { getEntry } from './registry'
import { useGlossaryStore } from './store'
import type { GlossaryId } from './types'

const CLASS = { text: 'gl-term', chip: 'embed-callout gl-term gl-term-chip', icon: 'gl-term-icon' } as const
// .gl-tip is at most 280px wide plus padding; keep it inside the viewport
const TIP_SPACE = 296

// A word in a caption (or a callout chip, or a heading's ⓘ) that shows the
// entry's summary on hover or focus and opens the entry on click.
export function Term({ id, section, variant = 'text', label, testId, children }: {
  id: GlossaryId
  section?: string
  variant?: 'text' | 'chip' | 'icon'
  label?: string
  testId?: string
  children: ReactNode
}) {
  const openEntry = useGlossaryStore((s) => s.openEntry)
  // viewport position of the tooltip, or null when hidden; fixed positioning
  // keeps it out of the drawer's scroll clipping
  const [tip, setTip] = useState<{ left: number; top: number } | null>(null)
  const tipId = useId()
  const entry = getEntry(id)
  if (!entry) return <span>{children}</span>
  return (
    <span className="gl-term-wrap">
      <button type="button" className={CLASS[variant]} data-testid={testId ?? 'gl-term'}
        data-term={id} data-section={section} aria-label={label}
        aria-describedby={tip ? tipId : undefined}
        onMouseEnter={(e) => setTip(place(e.currentTarget))} onMouseLeave={() => setTip(null)}
        onFocus={(e) => setTip(place(e.currentTarget))} onBlur={() => setTip(null)}
        onClick={(e) => { setTip(null); openEntry(entry.id, section, e.currentTarget) }}>
        {children}
      </button>
      {tip && (
        <span role="tooltip" id={tipId} data-testid="gl-tip" className="gl-tip" style={{ left: tip.left, top: tip.top }}>
          {entry.summary} <em>Click to read more.</em>
        </span>
      )}
    </span>
  )
}

function place(el: HTMLElement): { left: number; top: number } {
  const r = el.getBoundingClientRect()
  return { left: Math.max(8, Math.min(r.left, window.innerWidth - TIP_SPACE)), top: r.bottom + 4 }
}

import { useId, useState, type ReactNode } from 'react'
import { getEntry } from './registry'
import { useGlossaryStore } from './store'
import type { GlossaryId } from './types'

const CLASS = { text: 'gl-term', chip: 'embed-callout gl-term gl-term-chip', icon: 'gl-term-icon' } as const

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
  const [tip, setTip] = useState(false)
  const tipId = useId()
  const entry = getEntry(id)
  if (!entry) return <span>{children}</span>
  return (
    <span className="gl-term-wrap">
      <button type="button" className={CLASS[variant]} data-testid={testId ?? 'gl-term'}
        data-term={id} data-section={section} aria-label={label}
        aria-describedby={tip ? tipId : undefined}
        onMouseEnter={() => setTip(true)} onMouseLeave={() => setTip(false)}
        onFocus={() => setTip(true)} onBlur={() => setTip(false)}
        onClick={(e) => { setTip(false); openEntry(entry.id, section, e.currentTarget) }}>
        {children}
      </button>
      {tip && (
        <span role="tooltip" id={tipId} data-testid="gl-tip" className="gl-tip">
          {entry.summary} <em>Click to read more.</em>
        </span>
      )}
    </span>
  )
}

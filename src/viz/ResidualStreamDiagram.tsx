import { useState, type KeyboardEvent, type SVGProps } from 'react'
import { getEntry } from '../glossary/registry'
import { useGlossaryStore } from '../glossary/store'
import { Term } from '../glossary/Term'

type BoxId = 'attention' | 'mlp'
// left edge of each box in SVG units; the diagram is drawn at 1:1 scale
const TIP_X: Record<BoxId, number> = { attention: 150, mlp: 356 }

// The anatomy of one transformer layer: a single matrix (the residual
// stream) flows left to right; attention and the MLP each ADD their edits
// back into it. Repeated for every layer. The two boxes open their glossary
// entries (SVG cannot hold the HTML Term button, so they wire it up by hand).
export function ResidualStreamDiagram({ seqLen, dims, layers }: {
  seqLen: number
  dims: number
  layers: number
}) {
  const openEntry = useGlossaryStore((s) => s.openEntry)
  const [tip, setTip] = useState<BoxId | null>(null)
  const boxProps = (id: BoxId): SVGProps<SVGGElement> => ({
    role: 'button', tabIndex: 0, className: 'rs-hover gl-svg-term',
    'aria-label': `${getEntry(id)?.title}: open the glossary entry`,
    onClick: (e) => { setTip(null); openEntry(id, undefined, e.currentTarget) },
    onKeyDown: (e: KeyboardEvent<SVGGElement>) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setTip(null); openEntry(id, undefined, e.currentTarget) }
    },
    onMouseEnter: () => setTip(id), onMouseLeave: () => setTip(null),
    onFocus: () => setTip(id), onBlur: () => setTip(null),
  })
  return (
    <div data-testid="residual-diagram" className="residual-diagram">
      <svg width="560" height="118" viewBox="0 0 560 118" role="group"
        aria-label="residual stream: attention and MLP each add their output back into the same matrix">
        {/* the stream */}
        <line x1="8" y1="88" x2="530" y2="88" className="rs-stream" />
        <path d="M530 88 l-8 -5 v10 z" className="rs-arrowhead" />
        <text x="8" y="106" className="rs-shape">x [{seqLen}×{dims}]</text>
        <text x="470" y="106" className="rs-shape">x′ [{seqLen}×{dims}]</text>
        {/* + attention branch */}
        <path d="M120 88 v-30 h30" fill="none" className="rs-branch" />
        <g data-testid="rs-term-attention" {...boxProps('attention')}>
          <rect x="150" y="42" width="112" height="32" rx="6" className="rs-box" />
          <text x="206" y="62" textAnchor="middle" className="rs-box-label">attention</text>
        </g>
        <path d="M262 58 h30 v30" fill="none" className="rs-branch" />
        <circle cx="292" cy="88" r="7" className="rs-adder" />
        <text x="292" y="92" textAnchor="middle" className="rs-plus">+</text>
        <text x="206" y="30" textAnchor="middle" className="rs-note">mixes across tokens — heatmap rows are the weights</text>
        {/* + MLP branch */}
        <path d="M330 88 v-30 h26" fill="none" className="rs-branch" />
        <g data-testid="rs-term-mlp" {...boxProps('mlp')}>
          <rect x="356" y="42" width="88" height="32" rx="6" className="rs-box" />
          <text x="400" y="62" textAnchor="middle" className="rs-box-label">MLP</text>
        </g>
        <path d="M444 58 h26 v30" fill="none" className="rs-branch" />
        <circle cx="470" cy="88" r="7" className="rs-adder" />
        <text x="470" y="92" textAnchor="middle" className="rs-plus">+</text>
        <text x="400" y="30" textAnchor="middle" className="rs-note">each position independently</text>
      </svg>
      {tip && (
        <span role="tooltip" data-testid="gl-tip" className="gl-tip gl-tip-svg" style={{ left: TIP_X[tip], top: 78 }}>
          {getEntry(tip)?.summary} <em>Click to read more.</em>
        </span>
      )}
      <p className="rs-caption">
        × {layers} <Term id="layer">layers</Term> — every layer <em>adds</em> its <Term id="edit">edits</Term> onto
        the same matrix; the original embedding stays underneath.
      </p>
    </div>
  )
}

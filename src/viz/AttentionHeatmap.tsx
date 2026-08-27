import { useState } from 'react'
import type { AttentionHead, AttentionLabel, TokenInfo } from '../trace/types'

const HINTS: Record<AttentionLabel, string> = {
  'attention-sink': 'A bright first column: surplus attention parks on the first token as a learned “do nothing” default.',
  'previous-token': 'A bright diagonal: this head mostly copies from the token right before.',
  induction: 'On repeated tokens, this head looks at what followed the previous occurrence — the circuit behind in-context pattern completion.',
  coreference: 'Follow the pronoun’s row: it attends back to its antecedent.',
}

const CELL = 18
const LABEL_W = 64

export function AttentionHeatmap({ heads, tokens }: { heads: AttentionHead[]; tokens: TokenInfo[] }) {
  const [selected, setSelected] = useState(0)
  const head = heads[Math.min(selected, heads.length - 1)]
  if (!head) return null
  const n = head.matrix.length
  const label = (i: number) => tokens[i]?.text.trim() ?? `#${i}`

  return (
    <div data-testid="attention-heatmap" className="attention-heatmap">
      <div className="head-chip-row">
        {heads.map((h, i) => (
          <button key={`${h.layer}-${h.head}`} data-testid="head-chip" data-active={String(i === selected)}
            className="head-chip" onClick={() => setSelected(i)}>
            {h.label} <span className="head-loc">L{h.layer}·H{h.head}</span>
          </button>
        ))}
      </div>
      <svg width={LABEL_W + n * CELL + 4} height={n * CELL + 4} role="img"
        aria-label={`attention weights, ${head.label} head`}>
        {head.matrix.map((row, r) => (
          <g key={r}>
            <text x={LABEL_W - 6} y={r * CELL + CELL / 2 + 4} textAnchor="end" className="attn-label">
              {label(r)}
            </text>
            {row.map((w, c) => (
              <rect key={c} data-testid="attn-cell" x={LABEL_W + c * CELL} y={r * CELL}
                width={CELL - 1} height={CELL - 1}
                fill={`hsl(220 70% ${Math.round(96 - 56 * Math.min(1, w))}%)`}>
                <title>{`${label(r)} → ${label(c)}: ${Math.round(w * 100)}%`}</title>
              </rect>
            ))}
          </g>
        ))}
      </svg>
      <p data-testid="attn-hint" className="attn-hint">{HINTS[head.label]}</p>
      <p className="attn-note">Illustrative pattern (simulated) — real attention weights are not exposed by the browser model.</p>
    </div>
  )
}

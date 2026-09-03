import { useEffect, useState } from 'react'
import type { AttentionHead, AttentionLabel, TokenInfo } from '../trace/types'

const HINTS: Record<AttentionLabel, string> = {
  'attention-sink': 'A bright first column: surplus attention parks on the first token as a learned “do nothing” default.',
  'previous-token': 'A bright diagonal: this head mostly copies from the token right before.',
  induction: 'On repeated tokens, this head looks at what followed the previous occurrence — the circuit behind in-context pattern completion.',
  coreference: 'Follow the pronoun’s row: it points back at one earlier word. Check whether that word is the referent.',
  distinctive: 'Focused attention that fits no textbook pattern — look for what it tracks.',
  pinned: 'Hand-picked from the grid — compare against the patterns you know.',
}

const CELL = 20
const LABEL_W = 66
const DIAG_PAD = 58   // room above the matrix for the slanted column labels
const RIGHT_PAD = 70  // overhang of the right-most slanted label

export function AttentionHeatmap({ heads, tokens, focus }: {
  heads: AttentionHead[]; tokens: TokenInfo[]
  focus?: { layer: number; head: number; label: AttentionLabel }
}) {
  const [selected, setSelected] = useState(0)
  const [hovered, setHovered] = useState<{ r: number; c: number } | null>(null)
  const clampedSelected = Math.min(selected, heads.length - 1)
  const head = heads[clampedSelected]
  const focusKey = focus ? `${focus.layer}-${focus.head}-${focus.label}` : null

  useEffect(() => {
    if (!focusKey) return
    const i = heads.findIndex((h) => `${h.layer}-${h.head}-${h.label}` === focusKey)
    if (i >= 0) { setSelected(i); setHovered(null) }
    // heads is intentionally omitted: re-focus should only fire when the
    // requested (layer, head, label) triple changes, not on every re-render
    // (e.g. hover state changes) while heads stays the same reference.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusKey])

  if (!head) return null
  const n = head.matrix.length
  const label = (i: number) => tokens[i]?.text.trim() ?? `#${i}`

  return (
    <div data-testid="attention-heatmap" className="attention-heatmap">
      <div className="head-chip-row">
        {heads.map((h, i) => (
          <button key={`${h.layer}-${h.head}-${h.label}`} data-testid="head-chip" data-active={String(i === clampedSelected)}
            className="head-chip" onClick={() => { setSelected(i); setHovered(null) }}>
            {h.label} <span className="head-loc">L{h.layer}·H{h.head}</span>
            {h.score != null && <span className="head-score">· {h.score}</span>}
          </button>
        ))}
      </div>
      <svg width={LABEL_W + n * CELL + RIGHT_PAD} height={DIAG_PAD + n * CELL + 4}
        viewBox={`0 0 ${LABEL_W + n * CELL + RIGHT_PAD} ${DIAG_PAD + n * CELL + 4}`} role="img"
        aria-label={`attention weights, ${head.label} head`}>
        {head.matrix.map((_, i) => {
          const cx = LABEL_W + i * CELL + CELL / 2
          const cy = DIAG_PAD + i * CELL - 5
          return (
            <text key={`c${i}`} data-testid="col-label" data-hl={String(hovered?.c === i)}
              x={cx} y={cy} textAnchor="start" transform={`rotate(-45 ${cx} ${cy})`}
              className="attn-label attn-col-label">
              {label(i)}
            </text>
          )
        })}
        {head.matrix.map((row, r) => (
          <g key={r}>
            <text data-testid="row-label" data-hl={String(hovered?.r === r)}
              x={LABEL_W - 8} y={DIAG_PAD + r * CELL + CELL / 2 + 4} textAnchor="end" className="attn-label">
              {label(r)}
            </text>
            {row.map((w, c) => {
              const isHovered = hovered?.r === r && hovered?.c === c
              return (
                <rect key={c} data-testid="attn-cell" x={LABEL_W + c * CELL} y={DIAG_PAD + r * CELL}
                  width={CELL - 1} height={CELL - 1}
                  fill={`hsl(211 ${Math.round(30 + 25 * Math.min(1, w))}% ${Math.round(94 - 70 * Math.min(1, w))}%)`}
                  stroke={isHovered ? '#d64' : 'none'} strokeWidth={isHovered ? 2 : 0}
                  onMouseEnter={() => setHovered({ r, c })}
                  onMouseLeave={() => setHovered(null)}>
                  <title>{`${label(r)} → ${label(c)}: ${Math.round(w * 100)}%`}</title>
                </rect>
              )
            })}
          </g>
        ))}
      </svg>
      <p data-testid="attn-readout" className="attn-readout">
        {hovered
          ? `${label(hovered.r)} → ${label(hovered.c)}: ${Math.round(head.matrix[hovered.r][hovered.c] * 100)}%`
          : ' '}
      </p>
      <p data-testid="attn-hint" className="attn-hint">{HINTS[head.label]}</p>
      <p data-testid="attn-note" className="attn-note">
        {head.label === 'pinned' || head.score != null
          ? head.score != null
            ? 'Measured on this prompt — head roles detected from the attention weights, not labeled by the model.'
            : 'Measured on this prompt — attention accumulated over the whole run (the model finishes ahead of the replay).'
          : 'Illustrative pattern (simulated) — real attention weights are not exposed by the browser model.'}
      </p>
    </div>
  )
}

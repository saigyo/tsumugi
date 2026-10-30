import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { AttentionHead, AttentionLabel, ChatSpans, TokenInfo } from '../trace/types'
import { readableTokenText } from '../engine/tokenBytes'
import { visibleToken } from './tokenText'
import { LoadRealButton } from './LoadRealButton'
import { foldMatrix, type FoldedMatrix } from './foldMatrix'
import { cellSize, labelStep } from './heatmapSizing'

const HINTS: Record<AttentionLabel, string> = {
  'attention-sink': 'A bright first column: surplus attention parks on the first token as a learned “do nothing” default.',
  'previous-token': 'A bright diagonal: this head mostly copies from the token right before.',
  induction: 'On repeated tokens, this head looks at what followed the previous occurrence — the circuit behind in-context pattern completion.',
  coreference: 'Follow the pronoun’s row: it points back at one earlier word. Check whether that word is the referent.',
  distinctive: 'Focused attention that fits no textbook pattern — look for what it tracks.',
  pinned: 'Hand-picked from the grid — compare against the patterns you know.',
}

const LABEL_W = 66
const DIAG_PAD = 58   // room above the matrix for the slanted column labels
const RIGHT_PAD = 70  // overhang of the right-most slanted label

const Cells = memo(function Cells({ folded, cell, onHover, rowLabel, colLabel }: {
  folded: FoldedMatrix; cell: number; onHover: (h: { r: number; c: number } | null) => void
  rowLabel: (i: number) => string; colLabel: (j: number) => string
}) {
  return <>{folded.values.map((row, r) => row.map((w, c) => {
    const fold = folded.cols[c].to - folded.cols[c].from > 1
    return (
      <rect key={`${r}-${c}`} data-testid="attn-cell" data-row={r} data-col={c}
        x={LABEL_W + c * cell} y={DIAG_PAD + r * cell} width={cell - 1} height={cell - 1}
        fill={`hsl(211 ${Math.round(30 + 25 * Math.min(1, w))}% ${Math.round(94 - 70 * Math.min(1, w))}%)`}
        className={fold ? 'attn-cell-fold' : undefined}
        onMouseEnter={() => onHover({ r, c })} onMouseLeave={() => onHover(null)}>
        <title>{`${rowLabel(r)} → ${colLabel(c)}: ${Math.round(w * 100)}%`}</title>
      </rect>
    )
  }))}</>
})

export function AttentionHeatmap({ heads, tokens, focus, chat, foldToggle = true }: {
  heads: AttentionHead[]; tokens: TokenInfo[]
  chat?: ChatSpans; foldToggle?: boolean
  focus?: { layer: number; head: number; label: AttentionLabel }
}) {
  const [selected, setSelected] = useState(0)
  // ticked for one run only: a new run's spans fold the system prompt again
  const [shownFor, setShownFor] = useState<ChatSpans | null>(null)
  const showSystem = chat !== undefined && shownFor === chat
  const [hoveredRaw, setHovered] = useState<{ r: number; c: number } | null>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState<number | null>(null)
  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    setWidth(el.clientWidth || null)
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth || null))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
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

  const matrix = head?.matrix
  const system = chat?.system
  const folded = useMemo(
    () => matrix ? foldMatrix(matrix, showSystem && foldToggle ? undefined : system) : null,
    [matrix, system, showSystem, foldToggle])
  // compact labels: leading spaces dropped, newlines and tabs as glyphs
  const label = useCallback((i: number) => tokens[i] ? visibleToken(readableTokenText(tokens[i].id, tokens[i].text).replace(/^ +/, '')) : `#${i}`, [tokens])
  const rowLabel = useCallback((i: number) => folded ? label(folded.rows[i]) : '', [folded, label])
  const colLabel = useCallback((j: number) => {
    if (!folded) return ''
    const c = folded.cols[j]
    return c.to - c.from > 1 ? `system prompt (${c.to - c.from})` : label(c.from)
  }, [folded, label])

  if (!head || !folded) return null
  const nRows = folded.rows.length
  const nCols = folded.cols.length
  // a hover left over from a larger matrix points at nothing: drop it, so it
  // cannot come back if the matrix grows again
  const hovered = hoveredRaw && hoveredRaw.r < nRows && hoveredRaw.c < nCols ? hoveredRaw : null
  if (hoveredRaw && !hovered) setHovered(null)
  const cell = cellSize(width, nCols, LABEL_W + RIGHT_PAD)
  const step = labelStep(cell)

  return (
    <div ref={wrapRef} data-testid="attention-heatmap" className="attention-heatmap">
      <div className="head-chip-row">
        {heads.map((h, i) => (
          <button key={`${h.layer}-${h.head}-${h.label}`} data-testid="head-chip" data-active={String(i === clampedSelected)}
            className="head-chip" onClick={() => { setSelected(i); setHovered(null) }}>
            {h.label} <span className="head-loc">L{h.layer}·H{h.head}</span>
            {h.score != null && <span className="head-score">· {h.score}</span>}
          </button>
        ))}
      </div>
      {chat?.system && foldToggle && (
        <label className="attn-fold-toggle">
          <input type="checkbox" data-testid="attn-show-system" checked={showSystem}
            onChange={(e) => { setShownFor(e.target.checked ? chat ?? null : null); setHovered(null) }} />
          Show system prompt
        </label>
      )}
      <svg width={LABEL_W + nCols * cell + RIGHT_PAD} height={DIAG_PAD + nRows * cell + 4}
        viewBox={`0 0 ${LABEL_W + nCols * cell + RIGHT_PAD} ${DIAG_PAD + nRows * cell + 4}`} role="img"
        aria-label={`attention weights, ${head.label} head`}>
        {folded.cols.map((_, i) => {
          const cx = LABEL_W + i * cell + cell / 2
          const fold = folded.cols[i].to - folded.cols[i].from > 1
          // the folded column's first cell shares its row with the next column's:
          // its label starts half a row higher, midway between its neighbours'
          const firstRow = folded.rows.findIndex((r) => r >= folded.cols[i].from) - (fold ? 0.5 : 0)
          const cy = DIAG_PAD + firstRow * cell - 5
          return (
            <text key={`c${i}`} data-testid="col-label" data-hl={String(hovered?.c === i)}
              x={cx} y={cy} textAnchor="start" transform={`rotate(-45 ${cx} ${cy})`}
              className="attn-label attn-col-label">
              {i % step === 0 || hovered?.c === i ? colLabel(i) : ''}
            </text>
          )
        })}
        {folded.rows.map((_, r) => (
          <text key={`r${r}`} data-testid="row-label" data-hl={String(hovered?.r === r)}
            x={LABEL_W - 8} y={DIAG_PAD + r * cell + cell / 2 + 4} textAnchor="end" className="attn-label">
            {r % step === 0 || hovered?.r === r ? rowLabel(r) : ''}
          </text>
        ))}
        <Cells folded={folded} cell={cell} onHover={setHovered} rowLabel={rowLabel} colLabel={colLabel} />
        {hovered && (
          <rect data-testid="attn-hover" className="attn-hover" pointerEvents="none"
            x={LABEL_W + hovered.c * cell} y={DIAG_PAD + hovered.r * cell} width={cell - 1} height={cell - 1} />
        )}
      </svg>
      <p data-testid="attn-readout" className="attn-readout">
        {hovered
          ? `${rowLabel(hovered.r)} → ${colLabel(hovered.c)}: ${Math.round(folded.values[hovered.r][hovered.c] * 100)}%`
          : ' '}
      </p>
      <p data-testid="attn-hint" className="attn-hint">{HINTS[head.label]}{chat && head.label === 'induction' ? ' In Chat mode the template’s markers (<|im_start|>, ↵, <|im_end|>) repeat, and those repeats count too.' : ''}</p>
      <p data-testid="attn-note" className="attn-note">
        {head.label === 'pinned' || head.score != null
          ? head.score != null
            ? 'Measured on this prompt — head roles detected from the attention weights, not labeled by the model.'
            : 'Measured on this prompt — attention accumulated over the whole run (the model finishes ahead of the replay).'
          : <>Illustrative pattern (simulated): the real model would show the weights it computed. <LoadRealButton variant="link" /></>}
      </p>
    </div>
  )
}

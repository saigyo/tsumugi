import { useState, type ReactNode } from 'react'
import { EDIT_CALLOUTS } from '../glossary/callouts'
import { Term } from '../glossary/Term'
import { Formula } from './Formula'
import { FOLD_TITLE, foldItems, foldLabel, type FoldItem } from './chatFold'
import { ratioFill } from './mlpMath'
import type { ChatSpans } from '../trace/types'
import { useChatFold } from './chatFolds'
import type { EditRows } from './selectors'
import { visibleToken } from './tokenText'

type Block = 'attn' | 'mlp'
const BLOCK_NAME: Record<Block, string> = { attn: 'attention', mlp: 'MLP' }
// Bars stop at 2.0, like the heatmap's ≥ 1 colour: a first-layer edit of 12×
// would otherwise flatten every other bar. The title keeps the true value.
const BAR_CAP = 2
// A long prompt would draw tokens × 30 cells: show the most recent rows.
const MAX_ROWS = 64

// The last max token items; a fold item before the cut is dropped with it.
function capItems(items: FoldItem[], max: number): FoldItem[] {
  let tokens = 0
  for (let i = items.length - 1; i >= 0; i--)
    if (items[i].kind === 'token' && ++tokens === max) return items.slice(i)
  return items
}

// "How much each layer changes the vector": per token and layer, the size of
// each block's edit relative to the stream it is added to — one heatmap with
// an MLP / attention toggle, and bars for the selected token. Pure
// presentation; the block and the token are local view state.
export function EditSizeCard({ rows: all, note, noteAction, onLayer, chat }: { rows?: EditRows; note: string; noteAction?: ReactNode; onLayer?: (layer: number) => void; chat?: ChatSpans }) {
  const [block, setBlock] = useState<Block>('mlp')
  const [picked, setPicked] = useState<number | null>(null)
  const [open, setOpen] = useChatFold('edits', chat)
  const n = all ? all.tokens.length : 0
  const folded = all ? foldItems(n, chat?.system, open) : []
  const items = capItems(folded, MAX_ROWS)
  const shown = items.filter((it) => it.kind === 'token').length
  const last = n - 1
  // a pick hidden by the closed fold or the row cap falls back to the last token
  const sel = picked !== null && items.some((it) => it.kind === 'token' && it.index === picked) ? picked : last
  return (
    <section data-testid="edit-size-card" className="mlp-card">
      <h4>How much each layer changes the vector{' '}
        <Term id="edit" variant="icon" label="About edit sizes" testId="edit-size-info">ⓘ</Term></h4>
      {!all ? <p data-testid="edit-size-note" className="mlp-note">{note} {noteAction}</p> : (
        <>
          <p className="mlp-caption">Every block adds an <Term id="edit">edit</Term> to the{' '}
            <Term id="residual-stream">residual stream</Term>. The colour shows how big that edit is compared with
            the vector it is added to: 1.0 means the block wrote something as large as everything before it.</p>
          <Formula testId="edit-size-formula" className="card-formula" inline
            tex={String.raw`\displaystyle \text{size} = \frac{\lVert \Delta x \rVert}{\lVert x \rVert}`} />
          <div className="mlp-controls">
            <div className="mlp-seg" role="group" aria-label="Which block">
              {(['mlp', 'attn'] as const).map((b) => (
                <button key={b} type="button" data-testid="edit-block-toggle" aria-pressed={b === block}
                  onClick={() => setBlock(b)}>{BLOCK_NAME[b]}</button>
              ))}
            </div>
            <div data-testid="edit-legend" className="mlp-legend">0 <span className="mlp-ramp" aria-hidden="true" /> ≥ 1.0</div>
          </div>
          <EditHeatmap rows={all} items={items} open={open} onToggle={() => setOpen(!open)} block={block} selected={sel} onSelect={setPicked} onLayer={onLayer} />
          {shown < folded.filter((it) => it.kind === 'token').length && (
            <p data-testid="edit-rows-note" className="mlp-scale-note">
              showing the last {shown} of {n} tokens</p>
          )}
          <EditBars rows={all} index={sel} onLayer={onLayer} />
          <div className="embed-callouts">
            {EDIT_CALLOUTS.map((c) => (
              <Term key={c.label} id={c.id} section={c.section} tip={c.tip} variant="chip" testId="embed-callout">ⓘ {c.label}</Term>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

function EditHeatmap({ rows, items, open, onToggle, block, selected, onSelect, onLayer }: {
  rows: EditRows; items: FoldItem[]; open: boolean; onToggle: () => void; block: Block; selected: number; onSelect: (row: number) => void; onLayer?: (layer: number) => void
}) {
  const layers = rows[block][0]?.length ?? 0
  return (
    <div className="mlp-heatmap" role="group" aria-label={`${BLOCK_NAME[block]} edit sizes, tokens by layers`}>
      {items.map((item) => {
        if (item.kind === 'fold') return (
          <div key="fold" className="mlp-heatmap-row">
            <button type="button" data-testid="edit-fold" className="mlp-tok mlp-fold" aria-expanded={open}
              title={FOLD_TITLE} onClick={onToggle}>{foldLabel(item.count, open)}</button>
            <div className="mlp-cells mlp-cells-empty" />
          </div>
        )
        const r = item.index, t = rows.tokens[r]
        return (
        <div key={r} className="mlp-heatmap-row">
          <button type="button" data-testid="edit-row-label" aria-pressed={r === selected}
            className="mlp-tok" onClick={() => onSelect(r)}>{visibleToken(t)}</button>
          <div className="mlp-cells" style={{ gridTemplateColumns: `repeat(${layers}, 1fr)` }}>
            {rows[block][r].map((v, l) => (
              <div key={l} data-testid="edit-cell" data-block={block} data-row={r} data-layer={l}
                style={{ background: ratioFill(v) }} onClick={() => { onSelect(r); onLayer?.(l) }}
                title={`“${visibleToken(t)}”, layer ${l}, ${BLOCK_NAME[block]}: ${v.toFixed(2)}`} />
            ))}
          </div>
        </div>
        )
      })}
      <div className="mlp-heatmap-row">
        <span />
        <div data-testid="edit-axis" className="mlp-cells mlp-axis" style={{ gridTemplateColumns: `repeat(${layers}, 1fr)` }}>
          {Array.from({ length: layers }, (_, l) => <span key={l}>{l % 5 === 0 ? `L${l}` : ''}</span>)}
        </div>
      </div>
    </div>
  )
}

// The bar chart's frame, in viewBox units (the SVG stretches to the card).
const W = 900, H = 196, X0 = 40, Y0 = 164, TOP = 30

function EditBars({ rows, index, onLayer }: { rows: EditRows; index: number; onLayer?: (layer: number) => void }) {
  const a = rows.attn[index], m = rows.mlp[index]
  const layers = a.length
  const y = (v: number) => Y0 - (Math.min(v, BAR_CAP) / BAR_CAP) * (Y0 - TOP)
  const slot = (W - X0 - 8) / layers
  const bw = Math.max(1, (slot - 4) / 2)
  return (
    <figure className="mlp-chart-fig">
      <figcaption data-testid="edit-bars-label" className="mlp-chart-label">edit size per layer for
        “{visibleToken(rows.tokens[index])}”, attention vs MLP</figcaption>
      <svg data-testid="edit-bars" className="mlp-chart mlp-chart-wide" viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label="edit size per layer, attention and MLP bars">
        {[0, 0.5, 1, 1.5, 2].map((t) => (
          <g key={t}>
            <line x1={X0} x2={W - 6} y1={y(t)} y2={y(t)} data-testid={t === 1 ? 'edit-bars-ref' : undefined}
              className={t === 1 ? 'mlp-ref-line' : 'mlp-grid-line'} />
            <text x={X0 - 6} y={y(t) + 3} textAnchor="end" className="mlp-tick">{t.toFixed(1)}</text>
          </g>
        ))}
        <text x={W - 8} y={y(1) - 4} textAnchor="end" className="mlp-ref-label">edit as large as the stream itself</text>
        {a.map((_, l) => (['attn', 'mlp'] as const).map((block, j) => {
          const v = block === 'attn' ? a[l] : m[l]
          const capped = v > BAR_CAP
          return (
            <rect key={`${l}-${block}`} data-testid="edit-bar" data-block={block} data-layer={l} data-value={v.toFixed(2)}
              data-capped={String(capped)} className={`edit-bar edit-bar-${block}${capped ? ' edit-bar-capped' : ''}${onLayer ? ' edit-bar-clickable' : ''}`}
              onClick={() => onLayer?.(l)}
              x={X0 + l * slot + 2 + j * (bw + 1)} y={y(v)} width={bw} height={Y0 - y(v)}>
              <title>{`layer ${l}, ${BLOCK_NAME[block]}: ${v.toFixed(2)}`}</title>
            </rect>
          )
        }))}
        {a.map((_, l) => l % 5 === 0 && (
          <text key={l} x={X0 + l * slot + slot / 2} y={Y0 + 14} textAnchor="middle" className="mlp-tick">L{l}</text>
        ))}
        <rect x={X0 + 6} y={4} width={10} height={10} className="edit-bar-attn" />
        <text x={X0 + 20} y={13} className="mlp-tick">attention</text>
        <rect x={X0 + 90} y={4} width={10} height={10} className="edit-bar-mlp" />
        <text x={X0 + 104} y={13} className="mlp-tick">MLP</text>
        <text x={X0 + 150} y={13} className="mlp-tick">· bars stop at 2.0</text>
      </svg>
    </figure>
  )
}

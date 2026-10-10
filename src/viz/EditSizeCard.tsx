import { useState } from 'react'
import { EDIT_CALLOUTS } from '../glossary/callouts'
import { Term } from '../glossary/Term'
import { Formula } from './Formula'
import { ratioFill } from './mlpMath'
import type { EditRows } from './selectors'
import { visibleToken } from './tokenText'

const CELL = 7, ROW_H = 12, LABEL_W = 60, BAR_W = 3, BAR_H = 56
type Block = 'attn' | 'mlp'
const BLOCK_NAME: Record<Block, string> = { attn: 'attention', mlp: 'MLP' }

// "How much each layer changes the vector": per token and layer, the size of
// each block's edit relative to the stream it is added to. Pure
// presentation; the selected token is local (it only drives the bars).
export function EditSizeCard({ rows, note }: { rows?: EditRows; note: string }) {
  const [picked, setPicked] = useState<number | null>(null)
  const last = rows ? rows.tokens.length - 1 : -1
  const sel = picked !== null && picked <= last ? picked : last
  return (
    <section data-testid="edit-size-card" className="mlp-card">
      <h4>How much each layer changes the vector{' '}
        <Term id="edit" variant="icon" label="About edit sizes" testId="edit-size-info">ⓘ</Term></h4>
      {!rows ? <p data-testid="edit-size-note" className="mlp-note">{note}</p> : (
        <>
          <p className="mlp-caption">Every block adds an <Term id="edit">edit</Term> to the{' '}
            <Term id="residual-stream">residual stream</Term>. The colour shows how big that edit is compared with
            the vector it is added to: 1.0 means the block wrote something as large as everything before it.</p>
          <Formula testId="edit-size-formula" className="card-formula" inline
            tex={String.raw`\displaystyle \text{size} = \frac{\lVert \Delta x \rVert}{\lVert x \rVert}`} />
          <div className="edit-maps">
            {(['mlp', 'attn'] as const).map((block) => (
              <EditMap key={block} block={block} rows={rows} selected={sel} onSelect={setPicked} />
            ))}
          </div>
          <div className="mlp-scale">0 <span className="mlp-scale-ramp" aria-hidden="true" /> ≥ 1.0</div>
          <EditBars rows={rows} index={sel} />
          <div className="embed-callouts">
            {EDIT_CALLOUTS.map((c) => (
              <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="embed-callout">ⓘ {c.label}</Term>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

function EditMap({ block, rows, selected, onSelect }: {
  block: Block; rows: EditRows; selected: number; onSelect: (row: number) => void
}) {
  const layers = rows[block][0]?.length ?? 0
  const w = LABEL_W + layers * CELL, h = rows.tokens.length * ROW_H + 14
  return (
    <figure className="edit-map-fig">
      <figcaption>{BLOCK_NAME[block]}</figcaption>
      <svg data-testid="edit-map" data-block={block} width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="group"
        aria-label={`${BLOCK_NAME[block]} edit sizes, tokens by layers`}>
        {rows.tokens.map((t, r) => (
          <g key={r}>
            <text data-testid="edit-row-label" data-row={r} data-selected={String(r === selected)}
              className={`edit-row-label${r === selected ? ' edit-row-selected' : ''}`}
              x={LABEL_W - 4} y={r * ROW_H + 9} textAnchor="end" role="button" tabIndex={0}
              onClick={() => onSelect(r)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(r) }}>
              {visibleToken(t.text)}
            </text>
            {rows[block][r].map((v, l) => (
              <rect key={l} data-testid="edit-cell" data-block={block} data-row={r} data-layer={l}
                x={LABEL_W + l * CELL} y={r * ROW_H} width={CELL - 1} height={ROW_H - 1} fill={ratioFill(v)}
                onClick={() => onSelect(r)}>
                <title>{`“${visibleToken(t.text)}”, layer ${l}, ${BLOCK_NAME[block]}: ${v.toFixed(2)}`}</title>
              </rect>
            ))}
          </g>
        ))}
        {[0, 10, 20].filter((l) => l < layers).map((l) => (
          <text key={l} className="edit-axis" x={LABEL_W + l * CELL} y={h - 2}>L{l}</text>
        ))}
      </svg>
    </figure>
  )
}

function EditBars({ rows, index }: { rows: EditRows; index: number }) {
  const a = rows.attn[index], m = rows.mlp[index]
  const peak = Math.max(...a, ...m) || 1
  const w = a.length * (2 * BAR_W + 2)
  return (
    <figure className="edit-bars-fig">
      <figcaption data-testid="edit-bars-label">edit size per layer for “{visibleToken(rows.tokens[index].text)}”,
        attention vs MLP</figcaption>
      <svg width={w} height={BAR_H} viewBox={`0 0 ${w} ${BAR_H}`} role="img"
        aria-label="edit size per layer, attention and MLP bars">
        {a.map((_, l) => (['attn', 'mlp'] as const).map((block, j) => {
          const v = block === 'attn' ? a[l] : m[l]
          const bh = (v / peak) * (BAR_H - 2)
          return (
            <rect key={`${l}-${block}`} data-testid="edit-bar" data-block={block} data-layer={l} data-value={v.toFixed(2)}
              className={`edit-bar edit-bar-${block}`} x={l * (2 * BAR_W + 2) + j * BAR_W} y={BAR_H - bh}
              width={BAR_W - 0.5} height={bh}>
              <title>{`layer ${l}, ${BLOCK_NAME[block]}: ${v.toFixed(2)}`}</title>
            </rect>
          )
        }))}
      </svg>
    </figure>
  )
}

import { poolRow } from '../geometry/math'
import { logitsCallouts } from '../glossary/callouts'
import { Term } from '../glossary/Term'
import { READOUT_CELLS, type LogitCandidate } from '../trace/types'
import { thousands } from './selectors'
import { readableTokenText } from '../engine/tokenBytes'
import { visibleToken } from './tokenText'
import { rowMarkY } from './stripMath'
import { VectorStrip } from './VectorStrip'

const X_CELL_W = 5, X_CELL_H = 14          // bench strips (full card width available)
const EQ_CELL_W = 4                        // the x strip in the equation row, next to E and the scores
const DOT_CELL_W = 2, DOT_CELL_H = 10      // compact product strip per candidate
const BAR_W = 120


export interface LogitsReadoutProps {
  candidates: LogitCandidate[]
  lastHidden?: number[]
  dims: number
  vocabSize?: number
  selected: number
  onSelect: (index: number) => void
  assetRowFor: (id: number) => ArrayLike<number> | undefined   // PCA fallback rows
  assetNote: string                // shown when assetRowFor returns undefined
}

const sum = (a: number[]) => a.reduce((s, v) => s + v, 0)

// The Embeddings lookup read in reverse: x_last · Eᵀ = scores, then a bench
// that lays the selected candidate's dot product out cell by cell. Pure
// presentation; the container resolves vectors and owns the selection.
export function LogitsReadout({ candidates, lastHidden, dims, vocabSize, selected, onSelect, assetRowFor, assetNote }: LogitsReadoutProps) {
  const vocab = vocabSize ? thousands(vocabSize) : '?'
  const xCells = lastHidden ? poolRow(lastHidden, READOUT_CELLS) : undefined
  const per = Math.round(dims / READOUT_CELLS)
  const maxLogit = Math.max(...candidates.map((c) => c.logit))
  const sel = candidates[Math.min(selected, candidates.length - 1)]
  const exact = sel?.dot !== undefined && sel.row !== undefined && xCells !== undefined
  const assetVec = sel && !exact ? assetRowFor(sel.id) : undefined
  const assetCells = assetVec ? poolRow(assetVec, READOUT_CELLS) : undefined
  const dotSum = exact ? sum(sel.dot!) : undefined
  const missingX = <span data-testid="readout-x-missing" className="readout-x-missing">[1×{dims}] — not recorded in this run</span>
  return (
    <div data-testid="logits-readout" className="logits-readout">
      <div className="readout-equation">
        <div className="readout-x">
          <div className="embed-col-label">x_last [1×{dims}]</div>
          {xCells ? (
            <VectorStrip cells={xCells} cellW={EQ_CELL_W} cellH={X_CELL_H} testId="readout-x"
              ariaLabel="the last token’s final vector, pooled"
              titleFor={(c, v) => `dims ${c * per}–${c * per + per - 1}: mean ${v.toFixed(2)}`} />
          ) : missingX}
          <div className="embed-caption">the last position after the final layer and norm</div>
        </div>
        <div className="readout-op" aria-hidden="true">·</div>
        <svg data-testid="readout-matrix" className="embed-matrix" width="150" height="110" viewBox="0 0 150 110" role="img"
          aria-label={`the embedding matrix E read as Eᵀ, ${vocab} rows by ${dims} columns`}>
          <rect x="36" y="6" width="90" height="98" rx="4" className="rs-box" />
          <text x="81" y="24" textAnchor="middle" className="rs-box-label">Eᵀ</text>
          <text x="81" y="98" textAnchor="middle" className="rs-shape">[{dims} × {vocab}]</text>
          {candidates.map((c, i) => (
            <rect key={c.id} data-testid="readout-row-mark" x="36" y={rowMarkY(c.id)} width="90" height={i === selected ? 4 : 2}
              className={`readout-row-mark hue-fill-${i % 6}`} data-selected={String(i === selected)} />
          ))}
          <text x="4" y="60" className="rs-note">rows</text>
        </svg>
        <div className="readout-op" aria-hidden="true">=</div>
        <div className="readout-scores">
          <div className="embed-col-label">scores [{vocab}]</div>
          {candidates.map((c, i) => {
            const frac = Math.max(0.02, c.logit / maxLogit)
            return (
              <button key={c.id} type="button" data-testid="readout-candidate" data-selected={String(i === selected)}
                className="readout-candidate" onClick={() => onSelect(i)} title="Show this dot product on the bench">
                <span className={`token-chip hue-${i % 6}`}><span className="chip-text">{visibleToken(c)}</span></span>
                {c.dot ? (
                  <VectorStrip cells={c.dot} cellW={DOT_CELL_W} cellH={DOT_CELL_H} gap={0} testId="readout-product"
                    ariaLabel={`products for ${readableTokenText(c.id, c.text)}`} />
                ) : <span className="readout-nostrip" aria-hidden="true" />}
                <svg data-testid="logit-bar" data-token={c.text} width={BAR_W} height="14">
                  <rect width={BAR_W * frac} height="14" className="bar-rect" style={{ transition: 'width .4s' }} />
                </svg>
                <span className="bar-value">{c.logit.toFixed(1)}</span>
              </button>
            )
          })}
        </div>
      </div>

      {sel && (
        <div data-testid="readout-bench" className="readout-bench">
          <div className="embed-col-label">x_last ⊙ E[{visibleToken(sel)}]</div>
          <div className="readout-bench-rows">
            <span className="readout-bench-label">x_last</span>
            {xCells
              ? <VectorStrip cells={xCells} cellW={X_CELL_W} cellH={X_CELL_H} testId="bench-x" ariaLabel="x_last, pooled" />
              : <span className="readout-x-missing">[1×{dims}] — not recorded in this run</span>}
            <span className="readout-bench-label">E[{sel.id}]</span>
            {exact
              ? <VectorStrip cells={sel.row!} cellW={X_CELL_W} cellH={X_CELL_H} testId="bench-row" ariaLabel={`row ${sel.id} of E, pooled`} />
              : assetCells
                ? <VectorStrip cells={assetCells} cellW={X_CELL_W} cellH={X_CELL_H} testId="bench-row" ariaLabel={`row ${sel.id} of E, PCA-reduced`} />
                : <span className="embed-strip-missing">{assetNote}</span>}
            {exact && (
              <>
                <span className="readout-bench-label">product</span>
                <VectorStrip cells={sel.dot!} cellW={X_CELL_W} cellH={X_CELL_H} testId="bench-product"
                  ariaLabel="element-wise products, summed per cell"
                  titleFor={(c, v) => `dims ${c * per}–${c * per + per - 1}: x̄ ${xCells![c].toFixed(2)} · ē ${sel.row![c].toFixed(2)} → ${v.toFixed(2)}`} />
              </>
            )}
          </div>
          {exact ? (
            <div data-testid="readout-sum" className="readout-sum">
              Σ of the {READOUT_CELLS} cells = {dotSum!.toFixed(1)} — the logit on the bar
              {Math.abs(dotSum! - sel.logit) > 0.05 ? ` (logit ${sel.logit.toFixed(1)})` : ''}
            </div>
          ) : (
            <div data-testid="readout-sum" className="readout-sum">
              {assetCells ? `${assetCells.length} of ${dims} dimensions (PCA-reduced, offline). ` : ''}
              {lastHidden
                ? 'This candidate’s row was not among the 16 rows the model exported (a tie at the cut), so its products are not available.'
                : 'Run a real-mode generation to see the actual products.'}
            </div>
          )}
          {exact && (
            <div className="embed-caption">
              {READOUT_CELLS} cells of {per} dimensions; blue cells add to the score, red cells subtract.
            </div>
          )}
        </div>
      )}

      <div className="embed-callouts">
        {logitsCallouts(candidates.length, vocab).map((c) => (
          <Term key={c.label} id={c.id} section={c.section} tip={c.tip} variant="chip" testId="embed-callout">ⓘ {c.label}</Term>
        ))}
      </div>
    </div>
  )
}

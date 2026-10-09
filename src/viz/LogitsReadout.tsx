import { poolRow } from '../geometry/math'
import { READOUT_CELLS, type LogitCandidate } from '../trace/types'
import { thousands } from './selectors'
import { visibleToken } from './tokenText'
import { leadRatio } from './softmaxText'
import { rowMarkY } from './stripMath'
import { VectorStrip } from './VectorStrip'

const X_CELL_W = 5, X_CELL_H = 14          // bench strips (full card width available)
const EQ_CELL_W = 4                        // the x strip in the equation row, next to E and the scores
const DOT_CELL_W = 2, DOT_CELL_H = 10      // compact product strip per candidate
const BAR_W = 120

const SOFTMAX_CALLOUTS = (shown: number, vocab: string, T: number): Array<{ label: string; text: string }> => [
  { label: 'why exponentiate',
    text: `Scores can be negative and differ by arbitrary amounts. exp makes them all positive and turns a difference of d points into a ratio of e^(d / T), so the top score dominates in proportion to its lead: ${T > 0 ? `at T = ${T}, 1 point ahead is ${leadRatio(1, T)}× the probability, 2 points is ${leadRatio(2, T)}×` : 'at T = 0 the lead is all that counts'}.` },
  { label: 'temperature',
    text: 'Dividing every score by T before exponentiating scales every lead. T below 1 exaggerates leads (sharper, more predictable); T above 1 shrinks them (flatter, more random). T = 0 skips the draw and takes the top score: greedy decoding.' },
  { label: 'top-k only',
    text: `The sum in the denominator runs over the ${shown} candidates kept by top-k, so these percentages are shares among them. The other ${vocab} tokens minus ${shown} get nothing and can never be drawn.` },
]

const CALLOUTS = (shown: number, vocab: string): Array<{ label: string; text: string }> => [
  { label: 'tied with Embeddings',
    text: 'This is the Embeddings matrix read sideways: there a row was looked up by id; here every row is scored against x_last. Tied weights mean the model learns one vocabulary geometry, not two.' },
  { label: 'similar rows, similar scores',
    text: 'Tokens whose rows of E point the same way get nearly the same dot product, which is why near-synonyms sit together at the top.' },
  { label: `${vocab} scores, ${shown} shown`,
    text: 'The multiply produces one score per vocabulary entry; the trace keeps only the top-k the sampler can choose from.' },
]

export interface LogitsReadoutProps {
  candidates: LogitCandidate[]
  probs?: number[]                 // softmax cursor: one per candidate, same order
  temperature?: number             // with probs: the T the softmax used
  lastHidden?: number[]
  dims: number
  vocabSize?: number
  selected: number
  onSelect: (index: number) => void
  assetRowFor: (id: number) => ArrayLike<number> | undefined   // PCA fallback rows
  assetNote: string                // shown when assetRowFor returns undefined
}

const sum = (a: number[]) => a.reduce((s, v) => s + v, 0)

// One candidate's share, derived on its own numbers: lead below the top score,
// the relative weight exp(−lead / T), and the resulting percentage.
function shareDerivation(logit: number, top: number, T: number, prob: number): string {
  const pct = `${Math.round(prob * 100)}%`
  if (T === 0) return logit === top ? `T = 0: the top score takes everything → ${pct}` : `T = 0: not the top score → ${pct}`
  if (logit === top) return `the top score → weight 1.00 → ${pct}`
  const lead = top - logit
  const w = Math.exp(-lead / T)
  return `${lead.toFixed(1)} below the top score → exp(−${lead.toFixed(1)} / ${T}) = ${w.toFixed(2)} of the leader’s weight → ${pct}`
}

// The Embeddings lookup read in reverse: x_last · Eᵀ = scores, then a bench
// that lays the selected candidate's dot product out cell by cell. Pure
// presentation; the container resolves vectors and owns the selection.
export function LogitsReadout({ candidates, probs, temperature, lastHidden, dims, vocabSize, selected, onSelect, assetRowFor, assetNote }: LogitsReadoutProps) {
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
          <div className="embed-col-label">{probs ? 'softmax →' : 'scores'} [{vocab}]</div>
          {candidates.map((c, i) => {
            const frac = probs ? probs[i] : Math.max(0.02, c.logit / maxLogit)
            return (
              <button key={c.id} type="button" data-testid="readout-candidate" data-selected={String(i === selected)}
                className="readout-candidate" onClick={() => onSelect(i)} title="Show this dot product on the bench">
                <span className={`token-chip hue-${i % 6}`}><span className="chip-text">{visibleToken(c.text)}</span></span>
                {c.dot ? (
                  <VectorStrip cells={c.dot} cellW={DOT_CELL_W} cellH={DOT_CELL_H} gap={0} testId="readout-product"
                    ariaLabel={`products for ${c.text}`} />
                ) : <span className="readout-nostrip" aria-hidden="true" />}
                <svg data-testid="logit-bar" data-token={c.text} width={BAR_W} height="14">
                  {probs && temperature !== undefined && <title>{shareDerivation(c.logit, maxLogit, temperature, probs[i])}</title>}
                  <rect width={BAR_W * frac} height="14" className="bar-rect" style={{ transition: 'width .4s' }} />
                </svg>
                <span className="bar-value">{probs ? `${Math.round(probs[i] * 100)}%` : c.logit.toFixed(1)}</span>
              </button>
            )
          })}
        </div>
      </div>

      {sel && (
        <div data-testid="readout-bench" className="readout-bench">
          <div className="embed-col-label">x_last ⊙ E[{visibleToken(sel.text)}]</div>
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
              Run a real-mode generation to see the actual products.
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
        {(probs ? SOFTMAX_CALLOUTS(candidates.length, vocab, temperature ?? 1) : CALLOUTS(candidates.length, vocab)).map((c) => (
          <span key={c.label} data-testid="embed-callout" className="embed-callout rs-hover" title={c.text}>ⓘ {c.label}</span>
        ))}
      </div>
    </div>
  )
}

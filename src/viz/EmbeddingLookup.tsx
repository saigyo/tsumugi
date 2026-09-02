import { poolRow } from '../geometry/math'
import type { EmbedSource, TokenInfo } from '../trace/types'
import { thousands } from './selectors'
import { markLeadingSpace } from './spaceMarker'

const STRIP_CELLS = 96
const CELL_W = 5
const CELL_H = 14
// the stacked stream: each token's vector as a miniature strip
const STACK_CELLS = 48
const STACK_CELL_W = 2
const STACK_ROW_H = 4
const STACK_GAP = 2

const CALLOUTS: Array<{ label: string; text: string }> = [
  { label: 'learned, not designed',
    text: 'Every row starts random and is adjusted during training so that the rest of the network predicts well. Nobody chose what dimension 17 means.' },
  { label: 'no position here',
    text: 'SmolLM2 adds no position vector at this stage. Position enters inside attention, via rotary position embeddings applied to queries and keys.' },
  { label: 'tied with Logits',
    text: 'The same 49152 × 576 matrix is reused at the end: the final vector is compared against every row to score each token (tie_word_embeddings).' },
]

// Diverging fill: indigo for positive, vermillion for negative, intensity
// relative to the row's own peak magnitude.
function cellFill(v: number, peak: number): string {
  const t = Math.abs(v) / peak
  return v >= 0
    ? `hsl(211 45% ${Math.round(88 - t * 50)}%)`
    : `hsl(13 55% ${Math.round(90 - t * 40)}%)`
}

const peakOf = (cells: number[]) => cells.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1

// The mechanism half of the Embeddings card: ids → one row of E → the stacked
// residual stream x. Pure presentation; the container decides where vectors
// come from (trace rows or the geometry asset).
export function EmbeddingLookup({ tokens, dims, vocabSize, selected, onSelect, vectorFor, source, missingNote }: {
  tokens: TokenInfo[]
  dims: number
  vocabSize?: number
  selected: number
  onSelect: (pos: number) => void
  vectorFor: (pos: number) => ArrayLike<number> | undefined
  source: EmbedSource
  missingNote: string
}) {
  const token = tokens[selected]
  const vec = token ? vectorFor(selected) : undefined
  const cells = vec ? poolRow(vec, STRIP_CELLS) : []
  const peak = peakOf(cells)
  const vocab = vocabSize ? thousands(vocabSize) : '?'
  const rowY = token ? 40 + (token.id % 41) : 0   // a schematic position inside E
  const stackRows = tokens.map((_, i) => {
    const v = vectorFor(i)
    return v ? poolRow(v, STACK_CELLS) : null
  })
  const stackW = STACK_CELLS * STACK_CELL_W
  const stackH = tokens.length * (STACK_ROW_H + STACK_GAP)
  return (
    <div data-testid="embed-lookup" className="embed-lookup">
      <div className="embed-ids">
        <div className="embed-col-label">ids</div>
        <div className="token-chip-row embed-chips">
          {tokens.map((t, i) => (
            <button key={i} type="button" data-testid="embed-token" data-selected={String(i === selected)}
              className={`token-chip embed-chip hue-${i % 6}`} onClick={() => onSelect(i)}
              title={`Show row ${t.id} of E`}>
              <span className="chip-text">{markLeadingSpace(t.text)}</span>
              <span className="chip-id">{t.id}</span>
            </button>
          ))}
        </div>
      </div>
      <svg className="embed-matrix" width="150" height="110" viewBox="0 0 150 110" role="img"
        aria-label={`embedding matrix E, ${vocab} rows by ${dims} columns`}>
        <path d="M4 55 h26" className="rs-branch" />
        <path d="M32 55 l-7 -4 v8 z" className="rs-arrowhead" />
        <rect x="36" y="6" width="90" height="98" rx="4" className="rs-box" />
        <text x="81" y="24" textAnchor="middle" className="rs-box-label">E</text>
        <text x="81" y="98" textAnchor="middle" className="rs-shape">[{vocab} × {dims}]</text>
        {token && <rect x="36" y={rowY} width="90" height="4" className="embed-row-mark" />}
      </svg>
      <div className="embed-strip-wrap">
        <div className="embed-col-label">row {token?.id ?? '—'}</div>
        {cells.length > 0 ? (
          <svg data-testid="embed-strip" width={cells.length * CELL_W} height={CELL_H}
            viewBox={`0 0 ${cells.length * CELL_W} ${CELL_H}`} role="img"
            aria-label={`embedding row of token ${token?.text ?? ''}`}>
            {cells.map((v, c) => (
              <rect key={c} data-testid="embed-strip-cell" x={c * CELL_W} y={0} width={CELL_W - 1} height={CELL_H}
                fill={cellFill(v, peak)} />
            ))}
          </svg>
        ) : (
          <div data-testid="embed-strip-missing" className="embed-strip-missing">{missingNote}</div>
        )}
        {cells.length > 0 && (
          <div className="embed-caption">
            {source === 'model'
              ? `${dims} values, mean-pooled into ${STRIP_CELLS} cells; blue positive, red negative`
              : `${cells.length} of ${dims} dimensions (PCA-reduced, offline) — these are PCA components, ordered by variance, so the first cells are always the strongest; blue positive, red negative`}
          </div>
        )}
      </div>
      <div className="embed-stack">
        <div className="embed-col-label">x [{tokens.length} × {dims}]</div>
        <svg data-testid="embed-stack" className="embed-stack-rows" width={stackW} height={stackH}
          viewBox={`0 0 ${stackW} ${stackH}`} role="img"
          aria-label={`the residual stream: one row per token, ${tokens.length} rows`}>
          {stackRows.map((row, i) => {
            const y = i * (STACK_ROW_H + STACK_GAP)
            const rowPeak = row ? peakOf(row) : 1
            return (
              <g key={i} data-testid="embed-stack-row" data-newest={String(i === tokens.length - 1)}
                data-selected={String(i === selected)} data-filled={String(row !== null)}
                className="embed-stack-row">
                {row
                  ? row.map((v, c) => (
                    <rect key={c} data-testid="embed-stack-cell" x={c * STACK_CELL_W} y={y}
                      width={STACK_CELL_W} height={STACK_ROW_H} fill={cellFill(v, rowPeak)} />
                  ))
                  : <rect x={0} y={y} width={stackW} height={STACK_ROW_H} className="embed-stack-placeholder" />}
                <rect x={0.5} y={y + 0.5} width={stackW - 1} height={STACK_ROW_H - 1} className="embed-stack-frame" />
              </g>
            )
          })}
        </svg>
        <div className="embed-caption">
          one row per token, each a miniature of its vector; the newest was added this cycle
        </div>
      </div>
      <div className="embed-callouts">
        {CALLOUTS.map((c) => (
          <span key={c.label} data-testid="embed-callout" className="embed-callout rs-hover" title={c.text}>ⓘ {c.label}</span>
        ))}
      </div>
    </div>
  )
}

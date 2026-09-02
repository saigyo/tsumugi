import type { GeometryAsset } from '../geometry/asset'
import { renderableNeighbors, similarityMatrix } from '../geometry/math'
import type { EmbedSource, TokenInfo } from '../trace/types'
import { SPACE_MARKER, visibleToken } from './tokenText'

const NEIGHBORS = 8
const SPACE_NOTE = `${SPACE_MARKER} marks a leading space, ↵ a newline, ⇥ a tab: whitespace belongs to the token, so " cat" (mid-sentence) and "cat" (after punctuation or at a start) are different tokens with different rows in E, and a paragraph break is two "↵" tokens.`
const MATRIX_CAP = 24
const CELL = 18
const LABEL_W = 60
const DIAG_PAD = 52
const RIGHT_PAD = 60

// Smallest and largest off-diagonal value, or null when there is no spread
// to stretch (fewer than two tokens, or all pairs equal).
function offDiagonalRange(matrix: number[][] | null): { lo: number; hi: number } | null {
  if (!matrix || matrix.length < 2) return null
  let lo = Infinity, hi = -Infinity
  matrix.forEach((row, r) => row.forEach((v, c) => {
    if (r === c) return
    if (v < lo) lo = v
    if (v > hi) hi = v
  }))
  return hi > lo ? { lo, hi } : null
}

// The geometry half of the Embeddings card: nearest vocabulary neighbours of
// the selected token, and the visible sequence's self-similarity — what the
// model "knows" about these tokens before any context is applied.
export function EmbeddingGeometry({ tokens, selected, vectorFor, asset, loading, error, retry, canRetry = true, source }: {
  tokens: TokenInfo[]
  selected: number
  vectorFor: (pos: number) => ArrayLike<number> | undefined
  asset?: GeometryAsset
  loading: boolean
  error?: string
  retry: () => void
  canRetry?: boolean
  source: EmbedSource
}) {
  const token = tokens[selected]
  const start = Math.max(0, tokens.length - MATRIX_CAP)
  const shown = tokens.slice(start)
  const vectors = shown.map((_, i) => vectorFor(start + i))
  const matrix = vectors.every((v) => v !== undefined) ? similarityMatrix(vectors as ArrayLike<number>[]) : null
  // Embedding spaces are anisotropic: ordinary tokens all sit at cosine ~0.5
  // or above, so a fixed 0..1 scale paints one dark block. Stretch the colour
  // to this matrix's own off-diagonal range; the diagonal (1.0) stays darkest.
  const range = offDiagonalRange(matrix)
  const shade = (v: number, diagonal: boolean) => {
    const t = diagonal ? 1 : range ? Math.min(1, Math.max(0, (v - range.lo) / (range.hi - range.lo))) : Math.max(0, v)
    return `hsl(211 ${Math.round(30 + 25 * t)}% ${Math.round(94 - 70 * t)}%)`
  }
  const neighbors = asset && token ? renderableNeighbors(asset, token.id, NEIGHBORS) : null
  const label = (i: number) => visibleToken(shown[i]?.text ?? '') || `#${start + i}`
  const w = LABEL_W + shown.length * CELL + RIGHT_PAD
  const h = DIAG_PAD + shown.length * CELL + 4
  return (
    <div data-testid="embed-geometry" className="embed-geometry">
      <h4>Geometry: meaning is distance</h4>
      {loading && !asset && (
        <p data-testid="embed-geometry-loading" className="embed-caption">Loading vocabulary geometry…</p>
      )}
      {error && !asset && (
        <p data-testid="embed-geometry-error" className="notice">
          Vocabulary geometry couldn't be loaded{' '}
          {canRetry && (
            <button type="button" data-testid="embed-geometry-retry" className="explore-toggle" onClick={retry}>retry</button>
          )}
        </p>
      )}
      {neighbors && token && (
        <div data-testid="embed-neighbors" className="embed-neighbors">
          <div className="embed-caption">
            Nearest to <span className="chip-text">{visibleToken(token.text)}</span> in E — click any token above to change
            {' '}<span data-testid="embed-space-note" className="embed-callout rs-hover" title={SPACE_NOTE}>ⓘ {SPACE_MARKER}</span>
          </div>
          <div className="bar-chart">
            {neighbors.map((n) => (
              <div key={n.id} data-testid="embed-neighbor" className="bar-row">
                <span className="bar-label">{visibleToken(n.text)}</span>
                <svg width="120" height="12"><rect width={120 * n.sim} height="12" className="bar-rect" /></svg>
                <span className="bar-value">{n.sim.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {matrix && (
        <div className="embed-sim">
          <div className="embed-caption">
            Tokens vs. each other (cosine, before any context)
            {start > 0 && <span data-testid="embed-sim-cap"> — showing the last {MATRIX_CAP} of {tokens.length}</span>}
            {range && (
              <span data-testid="embed-sim-legend" className="rs-hover"
                title="Every pair of ordinary tokens scores high on cosine similarity (embedding spaces are anisotropic), so a fixed 0 to 1 scale would be one dark block. The colours span this matrix's own range; hover a cell for its exact value.">
                {' '}— colour range {range.lo.toFixed(2)} … {range.hi.toFixed(2)}, diagonal = 1
              </span>
            )}
          </div>
          <svg data-testid="embed-similarity" width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img"
            aria-label="cosine similarity between the embedding rows of the visible tokens">
            {shown.map((_, i) => {
              const cx = LABEL_W + i * CELL + CELL / 2
              const cy = DIAG_PAD - 5
              return (
                <text key={`c${i}`} x={cx} y={cy} textAnchor="start" transform={`rotate(-45 ${cx} ${cy})`}
                  className="attn-label">{label(i)}</text>
              )
            })}
            {matrix.map((row, r) => (
              <g key={r}>
                <text x={LABEL_W - 8} y={DIAG_PAD + r * CELL + CELL / 2 + 4} textAnchor="end" className="attn-label">
                  {label(r)}
                </text>
                {row.map((v, c) => {
                  return (
                    <rect key={c} data-testid="sim-cell" x={LABEL_W + c * CELL} y={DIAG_PAD + r * CELL}
                      width={CELL - 1} height={CELL - 1}
                      fill={shade(v, r === c)}>
                      <title>{`${label(r)} · ${label(c)}: ${v.toFixed(2)}`}</title>
                    </rect>
                  )
                })}
              </g>
            ))}
          </svg>
        </div>
      )}
      {(neighbors != null || matrix != null) && (
        <p data-testid="embed-provenance" className="embed-caption">
          {source === 'model'
            ? 'Exact rows from the running model.'
            : 'Real SmolLM2 embedding rows, reduced to 64 dimensions offline; similarities are approximate.'}
        </p>
      )}
    </div>
  )
}

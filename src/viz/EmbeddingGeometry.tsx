import type { GeometryAsset } from '../geometry/asset'
import { renderableNeighbors, similarityMatrix } from '../geometry/math'
import type { EmbedSource, TokenInfo } from '../trace/types'

const NEIGHBORS = 8
const MATRIX_CAP = 24
const CELL = 18
const LABEL_W = 60
const DIAG_PAD = 52
const RIGHT_PAD = 60

// The geometry half of the Embeddings card: nearest vocabulary neighbours of
// the selected token, and the visible sequence's self-similarity — what the
// model "knows" about these tokens before any context is applied.
export function EmbeddingGeometry({ tokens, selected, vectorFor, asset, loading, error, retry, source }: {
  tokens: TokenInfo[]
  selected: number
  vectorFor: (pos: number) => ArrayLike<number> | undefined
  asset?: GeometryAsset
  loading: boolean
  error?: string
  retry: () => void
  source: EmbedSource
}) {
  const token = tokens[selected]
  const start = Math.max(0, tokens.length - MATRIX_CAP)
  const shown = tokens.slice(start)
  const vectors = shown.map((_, i) => vectorFor(start + i))
  const matrix = vectors.every((v) => v !== undefined) ? similarityMatrix(vectors as ArrayLike<number>[]) : null
  const neighbors = asset && token ? renderableNeighbors(asset, token.id, NEIGHBORS) : null
  const label = (i: number) => shown[i]?.text.trim() || `#${start + i}`
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
          <button type="button" data-testid="embed-geometry-retry" className="explore-toggle" onClick={retry}>retry</button>
        </p>
      )}
      {neighbors && token && (
        <div data-testid="embed-neighbors" className="embed-neighbors">
          <div className="embed-caption">
            Nearest to <span className="chip-text">{token.text}</span> in E — click any token above to change
          </div>
          <div className="bar-chart">
            {neighbors.map((n) => (
              <div key={n.id} data-testid="embed-neighbor" className="bar-row">
                <span className="bar-label">{n.text}</span>
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
                  const m = Math.max(0, v)
                  return (
                    <rect key={c} data-testid="sim-cell" x={LABEL_W + c * CELL} y={DIAG_PAD + r * CELL}
                      width={CELL - 1} height={CELL - 1}
                      fill={`hsl(211 ${Math.round(30 + 25 * m)}% ${Math.round(94 - 70 * m)}%)`}>
                      <title>{`${label(r)} · ${label(c)}: ${v.toFixed(2)}`}</title>
                    </rect>
                  )
                })}
              </g>
            ))}
          </svg>
        </div>
      )}
      <p data-testid="embed-provenance" className="embed-caption">
        {source === 'model'
          ? 'Exact rows from the running model.'
          : 'Real SmolLM2 embedding rows, reduced to 64 dimensions offline; similarities are approximate.'}
      </p>
    </div>
  )
}

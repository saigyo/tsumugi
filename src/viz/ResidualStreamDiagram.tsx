// The anatomy of one transformer layer: a single matrix (the residual
// stream) flows left to right; attention and the MLP each ADD their edits
// back into it. Repeated for every layer.
export function ResidualStreamDiagram({ seqLen, dims, layers }: {
  seqLen: number
  dims: number
  layers: number
}) {
  return (
    <div data-testid="residual-diagram" className="residual-diagram">
      <svg width="560" height="118" viewBox="0 0 560 118" role="img"
        aria-label="residual stream: attention and MLP each add their output back into the same matrix">
        {/* the stream */}
        <line x1="8" y1="88" x2="530" y2="88" className="rs-stream" />
        <path d="M530 88 l-8 -5 v10 z" className="rs-arrowhead" />
        <text x="8" y="106" className="rs-shape">x [{seqLen}×{dims}]</text>
        <text x="470" y="106" className="rs-shape">x′ [{seqLen}×{dims}]</text>
        {/* + attention branch */}
        <path d="M120 88 v-30 h30" fill="none" className="rs-branch" />
        <rect x="150" y="42" width="112" height="32" rx="6" className="rs-box" />
        <text x="206" y="62" textAnchor="middle" className="rs-box-label">attention</text>
        <path d="M262 58 h30 v30" fill="none" className="rs-branch" />
        <circle cx="292" cy="88" r="7" className="rs-adder" />
        <text x="292" y="92" textAnchor="middle" className="rs-plus">+</text>
        <text x="206" y="30" textAnchor="middle" className="rs-note">mixes across tokens — heatmap rows are the weights</text>
        {/* + MLP branch */}
        <path d="M330 88 v-30 h26" fill="none" className="rs-branch" />
        <rect x="356" y="42" width="88" height="32" rx="6" className="rs-box" />
        <text x="400" y="62" textAnchor="middle" className="rs-box-label">MLP</text>
        <path d="M444 58 h26 v30" fill="none" className="rs-branch" />
        <circle cx="470" cy="88" r="7" className="rs-adder" />
        <text x="470" y="92" textAnchor="middle" className="rs-plus">+</text>
        <text x="400" y="30" textAnchor="middle" className="rs-note">each position independently</text>
      </svg>
      <p className="rs-caption">
        × {layers} layers — every layer <em>adds</em> its edits onto the same matrix;
        the original embedding stays underneath.
      </p>
    </div>
  )
}

// The SiLU curve for the MLP entry: near zero for negative inputs, close to
// the input for positive ones.
// SiLU(z) = z · σ(z), sampled for the small plot
const Z_MIN = -6, Z_MAX = 4, W = 220, H = 96, X0 = 30, Y0 = 74
const sx = (z: number) => X0 + ((z - Z_MIN) / (Z_MAX - Z_MIN)) * (W - X0 - 8)
const sy = (y: number) => Y0 - y * 16
const silu = (z: number) => z / (1 + Math.exp(-z))
const SILU_PATH = Array.from({ length: 101 }, (_, i) => {
  const z = Z_MIN + (i / 100) * (Z_MAX - Z_MIN)
  return `${i === 0 ? 'M' : 'L'}${sx(z).toFixed(1)} ${sy(silu(z)).toFixed(1)}`
}).join(' ')

export function SiluPlot() {
  return (
    <svg data-testid="gl-silu-plot" className="gl-plot" width={W} height={H} viewBox={`0 0 ${W} ${H}`}
      role="img" aria-label="SiLU: close to zero for negative inputs, close to the input itself for positive inputs">
      <line x1={X0} x2={W - 6} y1={sy(0)} y2={sy(0)} className="gl-plot-axis" />
      <line x1={sx(0)} x2={sx(0)} y1={6} y2={H - 6} className="gl-plot-axis" />
      <path d={SILU_PATH} className="gl-plot-curve" />
      <text x={W - 8} y={sy(0) + 12} textAnchor="end" className="gl-plot-label">z</text>
      <text x={sx(0) + 4} y={12} className="gl-plot-label">SiLU(z)</text>
      <text x={sx(-5)} y={sy(0) - 5} textAnchor="middle" className="gl-plot-label">≈ 0</text>
      <text x={sx(1.6)} y={sy(2.6)} textAnchor="middle" className="gl-plot-label">≈ z</text>
    </svg>
  )
}

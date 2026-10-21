import { memo, useMemo, type ReactNode } from 'react'
import type { productSummary } from '../neurons/steps'
import { silu } from '../neurons/steps'
import { cellFill } from './stripMath'
import { CHART_W, X0, X1, gateX, type Hover } from './stepChartMath'

// The step-by-step card's charts (#17). Every chart is an SVG on a 720-unit-wide
// frame drawn at width 100 %, so it fills the card's chart column (the #15
// sizing rule). The gate axis is shared: a neuron sits at the same x in the
// scatter and on the SiLU curve. Strips, dot layers and bars are memoized, so
// a hover change (new `hover`/`box` props) re-renders only the frame and the box;
// `onHover` must be stable for that (a state setter is).

const clamp = (v: number, lim: number) => Math.max(-lim, Math.min(lim, v))
const ticks = ([lo, hi]: [number, number]) => {
  const step = hi - lo > 12 ? 5 : hi - lo > 5 ? 2 : 1
  const out: number[] = []
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) out.push(t)
  return out
}

export const StepStrip = memo(function StepStrip({ values, lim, label, testId, labelClass }: { values: Float32Array; lim: number; label: string; testId: string; labelClass?: string }) {
  return (
    <div className="step-strip-row">
      <span className={labelClass ? `step-strip-label ${labelClass}` : 'step-strip-label'}>{label}</span>
      <svg data-testid={testId} className="step-strip" viewBox={`0 0 ${values.length} 1`} preserveAspectRatio="none" role="img" aria-label={label}>
        {Array.from(values, (v, i) => <rect key={i} x={i} y={0} width={1.02} height={1} fill={cellFill(clamp(v, lim), lim)} />)}
      </svg>
    </div>
  )
})

const Dots = memo(function Dots({ xs, ys, marked, chart, height, onHover }: { xs: number[]; ys: number[]; marked: number[]; chart: Hover['chart']; height: number; onHover(h: Hover | null): void }) {
  const m = new Set(marked)
  const show = (i: number) => onHover({ neuron: i, chart, x: xs[i], y: ys[i], height })
  const dot = (i: number) => (
    <circle key={i} data-neuron={i} cx={xs[i]} cy={ys[i]} r={m.has(i) ? 4 : 1.6}
      className={m.has(i) ? 'step-dot-marked' : 'step-dot'} tabIndex={m.has(i) ? 0 : undefined}
      role={m.has(i) ? 'img' : undefined} aria-label={m.has(i) ? `N${i}` : undefined}
      onMouseEnter={() => show(i)} onMouseLeave={() => onHover(null)}
      onFocus={() => show(i)} onBlur={() => onHover(null)} />
  )
  return <>{xs.map((_, i) => (m.has(i) ? null : dot(i)))}{marked.map(dot)}</>
})

function Frame({ testId, h, label, box, children }: { testId: string; h: number; label: string; box: ReactNode; children: ReactNode }) {
  return (
    <div className="step-chart">
      <svg data-testid={testId} className="mlp-chart" viewBox={`0 0 ${CHART_W} ${h}`} role="group" aria-label={label}>{children}</svg>
      {box}
    </div>
  )
}

export function GateUpScatter({ gate, up, domain, marked, named, hover, onHover, box }: {
  gate: Float32Array; up: Float32Array; domain: [number, number]; marked: number[]; named: number[]
  hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode
}) {
  const H = 210, top = 10, bottom = H - 20
  const { x, y, lo, hi, xs, ys } = useMemo(() => {
    const x = gateX(domain)
    let lo = -1, hi = 1
    for (const v of up) { if (v < lo) lo = v; if (v > hi) hi = v }
    const pad = (hi - lo) * 0.06
    const y = (v: number) => top + ((hi + pad - v) / (hi - lo + 2 * pad)) * (bottom - top)
    return { x, y, lo, hi, xs: Array.from(gate, x), ys: Array.from(up, y) }
  }, [gate, up, domain, bottom])
  return (
    <Frame testId="steps-scatter" h={H} label="gate against up, one dot per neuron" box={box}>
      <rect x={x(domain[0])} y={top} width={x(0) - x(domain[0])} height={bottom - top} className="step-squash" />
      <line x1={x(0)} x2={x(0)} y1={top} y2={bottom} className="step-zero" />
      <line x1={X0} x2={X1} y1={y(0)} y2={y(0)} className="step-zero" />
      <text x={x(domain[0]) + 6} y={top + 14} className="step-note">g &lt; 0: SiLU will squash these</text>
      <text x={X1 - 4} y={top + 14} textAnchor="end" className="step-axis">u ↑ · g →</text>
      {ticks(domain).map((t) => <text key={t} x={x(t)} y={H - 6} textAnchor="middle" className="step-axis">{t}</text>)}
      {ticks([lo, hi]).map((t) => <text key={t} x={X0 - 6} y={y(t) + 3} textAnchor="end" className="step-axis">{t}</text>)}
      <Dots xs={xs} ys={ys} marked={marked} chart="scatter" height={H} onHover={onHover} />
      <Ring hover={hover} xs={xs} ys={ys} />
      {named.map((i) => <text key={i} x={x(gate[i]) + 6} y={y(up[i]) + 4} className="step-name">N{i}</text>)}
    </Frame>
  )
}

export function SiluChart({ gate, domain, marked, counts, hover, onHover, box }: {
  gate: Float32Array; domain: [number, number]; marked: number[]; counts: { negative: number; positive: number }
  hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode
}) {
  const H = 150, top = 8, bottom = H - 20
  const { x, y, yMax, curve, xs, ys } = useMemo(() => {
    const x = gateX(domain)
    const yMax = Math.max(1, silu(domain[1]))
    const y = (v: number) => top + ((yMax - v) / (yMax + 0.6)) * (bottom - top)
    const curve = Array.from({ length: 241 }, (_, k) => domain[0] + (k / 240) * (domain[1] - domain[0]))
      .map((v) => `${x(v).toFixed(1)},${y(silu(v)).toFixed(1)}`).join(' ')
    return { x, y, yMax, curve, xs: Array.from(gate, x), ys: Array.from(gate, (g) => y(silu(g))) }
  }, [gate, domain, bottom])
  // The positive-gates note sits above and left of the curve, where the curve is lower than the text (as in the mockup).
  const noteX = Math.min(X1 - 4, Math.max(x(0) + 150, x(domain[1] * 0.55)))
  const noteG = domain[0] + ((noteX - X0) / (X1 - X0)) * (domain[1] - domain[0])
  const noteY = Math.max(top + 30, y(silu(noteG)) - 24)
  return (
    <Frame testId="steps-silu" h={H} label="SiLU of the gate, one dot per neuron" box={box}>
      <rect x={x(domain[0])} y={top} width={x(0) - x(domain[0])} height={bottom - top} className="step-squash" />
      <line x1={X0} x2={X1} y1={y(0)} y2={y(0)} className="step-zero" />
      <line x1={x(0)} x2={x(0)} y1={top} y2={bottom} className="step-zero" />
      <polyline points={curve} className="step-curve" />
      <text x={x(domain[0]) + 6} y={y(0) - 22} className="step-note">{counts.negative} negative gates</text>
      <text x={x(domain[0]) + 6} y={y(0) - 10} className="step-note">→ between −0.28 and 0</text>
      <text x={Math.min(x(0) + 6, X1 - 84)} y={top + 12} className="step-axis">SiLU ↑ · g →</text>
      <text x={noteX} y={noteY} textAnchor="end" className="step-note">{counts.positive} positive gates pass</text>
      <text x={noteX} y={noteY + 12} textAnchor="end" className="step-note">almost unchanged</text>
      {ticks(domain).map((t) => <text key={t} x={x(t)} y={H - 6} textAnchor="middle" className="step-axis">{t}</text>)}
      {ticks([0, yMax]).map((t) => <text key={t} x={X0 - 6} y={y(t) + 3} textAnchor="end" className="step-axis">{t}</text>)}
      <Dots xs={xs} ys={ys} marked={marked} chart="silu" height={H} onHover={onHover} />
      <Ring hover={hover} xs={xs} ys={ys} />
    </Frame>
  )
}

// The neuron hovered in any step, ringed in every step so it can be followed down the card.
function Ring({ hover, xs, ys }: { hover: Hover | null; xs: number[]; ys: number[] }) {
  if (!hover) return null
  const i = hover.neuron
  return <circle data-testid="steps-ring" data-neuron={i} cx={xs[i]} cy={ys[i]} r={8} className="step-ring" />
}

const Bars = memo(function Bars({ h, order, peak, marked, mid, reach, height, onHover }: {
  h: Float32Array; order: number[]; peak: number; marked: number[]; mid: number; reach: number; height: number; onHover(h: Hover | null): void
}) {
  const bw = (X1 - X0) / Math.max(order.length, 1)
  const m = new Set(marked)
  return <>{order.map((i, k) => {
    const x = X0 + k * bw
    const len = (Math.abs(h[i]) / peak) * reach
    const hv = { neuron: i, chart: 'product' as const, x: x + bw / 2, y: h[i] >= 0 ? mid - len : mid + len, height }
    return <rect key={i} data-neuron={i} x={x} width={Math.max(bw - 1, 0.6)} y={h[i] >= 0 ? mid - len : mid} height={len}
      className={m.has(i) ? 'step-bar-marked' : h[i] >= 0 ? 'step-bar-pos' : 'step-bar-neg'}
      tabIndex={m.has(i) ? 0 : undefined} role={m.has(i) ? 'img' : undefined} aria-label={m.has(i) ? `N${i}` : undefined}
      onMouseEnter={() => onHover(hv)} onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(hv)} onBlur={() => onHover(null)} />
  })}</>
})

export function ProductBars({ h, summary, marked, hover, onHover, box }: {
  h: Float32Array; summary: ReturnType<typeof productSummary>; marked: number[]
  hover: Hover | null; onHover(h: Hover | null): void; box: ReactNode
}) {
  const H = 150, mid = H / 2, reach = mid - 14
  const { order, half, ninety, restMax } = summary
  const peak = Math.abs(h[order[0]] ?? 0) || 1
  const bw = (X1 - X0) / Math.max(order.length, 1)
  const at = (k: number) => X0 + k * bw
  // A count within the bars gets a cut line; one beyond them a note at the right end.
  const cuts: Array<[number, string, number]> = [[half, 'half', 16], [ninety, '90 %', 30]]
  return (
    <Frame testId="steps-product" h={H} label="the largest activations, largest first" box={box}>
      <line x1={X0} x2={X1} y1={mid} y2={mid} className="step-zero" />
      <Bars h={h} order={order} peak={peak} marked={marked} mid={mid} reach={reach} height={H} onHover={onHover} />
      {hover && order.includes(hover.neuron) && (
        <rect data-testid="steps-ring" data-neuron={hover.neuron} className="step-ring" rx={3}
          x={at(order.indexOf(hover.neuron)) + bw / 2 - 5} width={10} y={4} height={H - 8} />)}
      {cuts.map(([k, name, ty]) => k > 0 && (k <= order.length
        ? <g key={name}>
            <line x1={at(k)} x2={at(k)} y1={6} y2={H - 6} className="step-cut" />
            <text x={at(k) + 3} y={ty} className="step-note">{k} {k === 1 ? 'neuron carries' : 'neurons carry'} {name}</text>
          </g>
        : <text key={name} x={X1 - 4} y={ty} textAnchor="end" className="step-note">{name} needs {k} neurons</text>))}
      <text x={X0 - 6} y={mid - reach + 4} textAnchor="end" className="step-axis">+{peak.toFixed(0)}</text>
      <text x={X0 - 6} y={mid + reach} textAnchor="end" className="step-axis">−{peak.toFixed(0)}</text>
      <text x={X1} y={H - 6} textAnchor="end" className="step-axis">
        the {order.length} largest, largest first →{restMax !== null && ` · the other ${h.length - order.length} are all below ${restMax < 0.1 ? restMax.toFixed(2) : restMax.toFixed(1)}`}</text>
    </Frame>
  )
}

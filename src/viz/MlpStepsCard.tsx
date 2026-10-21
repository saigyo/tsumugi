import { memo, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { MLP_STEP_CALLOUTS } from '../glossary/callouts'
import { Term } from '../glossary/Term'
import type { NeuronAsset } from '../neurons/asset'
import type { NeuronLayer } from '../neurons/format'
import { columnNorm, layerActivations, listedNeurons, neuronLabels } from '../neurons/math'
import { contribution, decodeSteps, editRatio, gateCounts, gateDomain, layerSteps, normalize, productSummary, rms, silu, type LayerSteps } from '../neurons/steps'
import { useNeuronLayer } from '../neurons/useNeuronLayer'
import { unpackFloat32, unpackHalf } from '../trace/neuronPack'
import type { NeuronData, TraceEvent } from '../trace/types'
import { Formula } from './Formula'
import { GateUpScatter, ProductBars, SiluChart, StepStrip } from './MlpStepCharts'
import { CHART_W, type Hover } from './stepChartMath'
import { useMlpLayerControl } from './mlpLayerSelection'
import { useNeuronMode } from './neuronMode'
import type { VotedCandidate } from './selectors'
import { visibleToken } from './tokenText'

const LOADING = 'Loading the MLP weights (about 0.9 MB per layer, once per model version)…'
const MISMATCH = 'This run’s step data doesn’t match the MLP weights file.'
const INCONSISTENT = 'This run’s step data is inconsistent and can’t be shown.'
const f2 = (v: number) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}`

const acts = new WeakMap<NeuronData, Float32Array>()
const activationsOf = (d: NeuronData) => { let a = acts.get(d); if (!a) { a = unpackHalf(d.activations); acts.set(d, a) } return a }

// "The MLP, step by step" (#17): one layer's MLP for the last token in five
// steps, with the model's own numbers. Layer and marked neurons follow
// "Inside one MLP"; steps 1 and 5 need the layer file (norm weight, columns).
export function MlpStepsCard({ data, candidates, note, noteAction, run }: {
  data?: NeuronData; candidates?: VotedCandidate[]; note: string; noteAction?: ReactNode; run?: TraceEvent
}) {
  const ready = data?.steps && candidates && candidates.length > 0
  const { layer, layers, selected, step } = useMlpLayerControl(ready ? candidates : undefined, run)
  return (
    <section data-testid="steps-card" className="mlp-card">
      <h4 data-testid="steps-title">The MLP, step by step{layer !== null ? `: layer ${layer}` : ''}{layer !== null && (
        <span className="neuron-stepper">
          <button type="button" data-testid="steps-prev" aria-label="Previous layer" disabled={layer <= 0} onClick={() => step(-1)}>◀</button>
          <button type="button" data-testid="steps-next" aria-label="Next layer" disabled={layer >= layers - 1} onClick={() => step(1)}>▶</button>
        </span>
      )}{' '}<Term id="mlp" variant="icon" label="About the MLP" testId="steps-info">ⓘ</Term></h4>
      {!ready || layer === null ? <p data-testid="steps-note" className="mlp-note">{note} {noteAction}</p>
        : <StepsBody data={data} candidates={candidates} selected={selected} layer={layer} />}
    </section>
  )
}

function StepsBody({ data, candidates, selected, layer }: { data: NeuronData; candidates: VotedCandidate[]; selected: number; layer: number }) {
  const state = useNeuronLayer(layer)
  const mode = useNeuronMode((s) => s.mode)
  const [hover, setHover] = useState<Hover | null>(null)
  const dims = decodeSteps(data.steps!).mlpIn.length / (activationsOf(data).length / data.neurons)
  const s = useMemo(() => Number.isInteger(dims) ? layerSteps(decodeSteps(data.steps!), activationsOf(data), layer, data.neurons, dims) : null,
    [data, layer, dims])
  const fits = state.status === 'ready' && state.asset.manifest.neurons === data.neurons && state.asset.manifest.dims === dims
  // useNeuronLayer returns a new object every render: key the memos on its stable parts
  const wAsset = fits ? state.asset : null, wLayer = fits ? state.layer : null
  const weights = useMemo(() => wAsset && wLayer ? { asset: wAsset, layer: wLayer } : null, [wAsset, wLayer])
  const rowB64 = data.candidates.find((c) => c.id === candidates[selected].id)?.row
  const marked = useMemo(() => {
    if (!weights || !s) return []
    const row = mode === 'vote' && rowB64 ? unpackFloat32(rowB64) : null
    return listedNeurons(weights.asset.manifest, weights.layer, layerActivations(activationsOf(data), layer, data.neurons), mode, row, weights.asset.g, data.invRms)
  }, [weights, s, mode, rowB64, data, layer])
  // per-layer summaries: computed once per layer, not on every hover
  const sum = useMemo(() => s && { domain: gateDomain(s.gate), counts: gateCounts(s.gate), product: productSummary(s.h) }, [s])
  const named = useMemo(() => s ? [...marked].sort((a, b) => Math.abs(s.h[b]) - Math.abs(s.h[a])).slice(0, 3) : [], [s, marked])
  if (!s || !sum) return <p data-testid="steps-note" className="mlp-note">{INCONSISTENT}</p>
  const box = (chart: Hover['chart']) => hover?.chart === chart
    ? <HoverBox s={s} hover={hover} weights={weights} /> : null
  const pending = state.status === 'ready'
    ? <p data-testid="steps-mismatch" className="mlp-note">{MISMATCH}</p>
    : state.status === 'error'
    ? <p data-testid="steps-error" className="mlp-note">Couldn’t load the MLP weights.{' '}
        <button type="button" data-testid="steps-retry" onClick={state.retry}>Retry</button></p>
    : <p data-testid="steps-loading" className="mlp-note">{LOADING}</p>
  return (
    <>
      <p className="mlp-caption">What layer {layer}’s MLP computes for the last token, in the order it computes it. Same layer as the
        cards above; <span className="step-mark" aria-hidden="true" /> marks the neurons listed in “Inside one MLP”.</p>
      <Step n={1} name="Norm" tex={String.raw`n = \operatorname{norm}(x)`}
        text="The vector coming into the MLP is scaled to a standard size, then each of the 576 numbers gets its own learned weight w.">
        {weights ? <NormStrips s={s} w={weights.layer.normWeight} eps={weights.asset.manifest.rmsNormEps} /> : pending}
      </Step>
      <Step n={2} name="Gate and up" tex={String.raw`g = W_{\text{gate}} \cdot n \quad u = W_{\text{up}} \cdot n`}
        text="Two different weight matrices turn n into two lists of 1,536 numbers, one pair per neuron. Each dot is one neuron.">
        <GateUpScatter gate={s.gate} up={s.up} domain={sum.domain} marked={marked} named={named} hover={hover} onHover={setHover} box={box('scatter')} />
        {named.length > 0 && <p data-testid="steps-scatter-note" className="mlp-scale-note">{namedNote(named.length)}</p>}
      </Step>
      <Step n={3} name="SiLU" tex={String.raw`\operatorname{SiLU}(g) = g \cdot \sigma(g)`}
        text="The gate goes through SiLU. Same gate axis as above: each dot slides straight down onto the curve.">
        <SiluChart gate={s.gate} domain={sum.domain} marked={marked} counts={sum.counts} hover={hover} onHover={setHover} box={box('silu')} />
      </Step>
      <Step n={4} name="Product" tex={String.raw`h = \operatorname{SiLU}(g) \odot u`}
        text="Each neuron’s up value is scaled by its gate. This is the neuron’s activation, the number “Inside one MLP” shows.">
        <ProductBars h={s.h} summary={sum.product} marked={marked} hover={hover} onHover={setHover} box={box('product')} />
      </Step>
      <Step n={5} name="Down" tex={String.raw`\text{edit} = W_{\text{down}} \cdot h`}
        text={<>Each neuron adds its column of W<sub>down</sub>, scaled by its activation. The sum is what this layer’s MLP adds to the stream.</>}>
        {weights ? <DownStrips s={s} layer={weights.layer} dims={dims} marked={marked} ring={hover?.neuron ?? null} /> : pending}
      </Step>
      <div className="embed-callouts">
        {MLP_STEP_CALLOUTS.map((c) => <Term key={c.label} id={c.id} section={c.section} tip={c.tip} variant="chip" testId="steps-callout">ⓘ {c.label}</Term>)}
      </div>
    </>
  )
}

function Step({ n, name, tex, text, children }: { n: number; name: string; tex: string; text: ReactNode; children: ReactNode }) {
  return (
    <div className="step-row">
      <div><h5>{n} · {name}</h5><Formula inline tex={tex} /><p className="step-text">{text}</p></div>
      <div className="step-charts">{children}</div>
    </div>
  )
}

const NormStrips = memo(function NormStrips({ s, w, eps }: { s: LayerSteps; w: Float32Array; eps: number }) {
  const { n, rms: r } = useMemo(() => normalize(s.x, w, eps), [s.x, w, eps])
  const peakW = useMemo(() => w.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1, [w])
  return (
    <div>
      <StepStrip values={s.x} lim={2 * rms(s.x)} label="x, into the MLP" testId="strip-x" />
      <StepStrip values={w} lim={peakW} label={`÷ rms(x) = ${r.toFixed(1)}, ⊙ w`} testId="strip-w" />
      <StepStrip values={n} lim={2 * rms(n)} label="n, normalized" testId="strip-n" />
      <p className="mlp-scale-note">{s.x.length} numbers each; blue positive, red negative; x and n clipped at 2× their typical size, w scaled to its largest value.</p>
    </div>
  )
})

const DownStrips = memo(function DownStrips({ s, layer, dims, marked, ring }: { s: LayerSteps; layer: NeuronLayer; dims: number; marked: number[]; ring: number | null }) {
  const top = useMemo(() => [...marked].map((i) => ({ i, size: Math.abs(s.h[i]) * columnNorm(layer, dims, i) }))
    .sort((a, b) => b.size - a.size).slice(0, 3).map(({ i }) => ({ i, values: contribution(layer, dims, i, s.h[i]) })), [marked, s, layer, dims])
  const lim = 2 * rms(s.edit)
  return (
    <div>
      {/* the wrapper carries the ring, so a hover leaves the strip itself (and its cells) untouched */}
      {top.map(({ i, values }) => <div key={i} className={i === ring ? 'step-row-ring' : undefined}>
        <StepStrip values={values} lim={lim} label={`N${i}: ${f2(s.h[i])} × column`} testId={`strip-c${i}`} labelClass="step-strip-mono" /></div>)}
      <div className="step-strip-row"><span className="step-strip-label">+ {s.h.length - top.length} more</span><span className="mlp-note">…</span></div>
      <StepStrip values={s.edit} lim={lim} label="= edit" testId="strip-edit" labelClass="step-strip-sum" />
      <p data-testid="steps-ratio" className="mlp-scale-note">‖edit‖ = {Math.hypot(...s.edit).toFixed(0)},{' '}
        {editRatio(s.edit, s.x).toFixed(2)} of the stream’s length (the edit size card’s cell). Same colour scale on every row.</p>
    </div>
  )
})

// Placed next to the hovered mark, as in the mockup: to its left in the right half of the chart, to its right otherwise;
// above it in the lower half, below it otherwise, so it stays inside the chart cell (which clips overflow).
function HoverBox({ s, hover, weights }: { s: LayerSteps; hover: Hover; weights: { asset: NeuronAsset; layer: NeuronLayer } | null }) {
  const i = hover.neuron
  const left = hover.x > CHART_W / 2
  const up = hover.y > hover.height / 2
  const style = { left: `${(hover.x / CHART_W) * 100}%`, top: `${(hover.y / hover.height) * 100}%`,
    transform: `translate(${left ? 'calc(-100% - 10px)' : '10px'}, ${up ? 'calc(-100% - 8px)' : '8px'})` }
  const labels = weights ? neuronLabels(weights.asset.manifest, weights.layer, i, s.h[i] >= 0 ? 0 : 1).slice(0, 3).map((id) => visibleToken(weights.asset.text(id))) : []
  // A mark near the middle (a short bar in step 4) can leave too little room either way: nudge the box
  // back inside the chart frame. Uses the separate `translate` property so the transform above stays as is.
  const ref = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const el = ref.current, frame = el?.parentElement
    if (!el || !frame) return
    el.style.translate = ''
    const b = el.getBoundingClientRect(), f = frame.getBoundingClientRect()
    const dy = b.top < f.top ? f.top - b.top : b.bottom > f.bottom ? Math.max(f.bottom - b.bottom, f.top - b.top) : 0
    if (dy) el.style.translate = `0 ${dy}px`
  }, [hover, labels.length])
  return (
    <div ref={ref} data-testid="steps-box" className="step-box" role="status" style={style}>
      <b>N{i}</b>
      <span>g {f2(s.gate[i])} → SiLU {f2(silu(s.gate[i]))}</span>
      <span>× u {f2(s.up[i])} = h {f2(s.h[i])}</span>
      {labels.length > 0 && <span className="step-box-labels">pushes toward {labels.join(', ')}…</span>}
    </div>
  )
}

// Only the largest marked neurons carry numbers in the scatter: the rest cluster where labels would overlap.
function namedNote(n: number): string {
  const which = n === 1 ? 'the marked neuron with the largest activation'
    : `the ${['', '', 'two', 'three'][n] ?? n} marked neurons with the largest activations`
  return `Numbers: ${which}. Hover or focus any dot for its number.`
}

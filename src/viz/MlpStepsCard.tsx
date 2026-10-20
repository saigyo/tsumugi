import { useMemo, useState, type ReactNode } from 'react'
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
const f2 = (v: number) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}`

const acts = new WeakMap<NeuronData, Float32Array>()
const activationsOf = (d: NeuronData) => { let a = acts.get(d); if (!a) { a = unpackHalf(d.activations); acts.set(d, a) } return a }

// "The MLP, step by step" (#17): one layer's MLP for the last token in five
// steps, with the model's own numbers. Layer and marked neurons follow
// "Inside one MLP"; steps 1 and 5 need the layer file (norm weight, columns).
export function MlpStepsCard({ data, candidates, note, run }: {
  data?: NeuronData; candidates?: VotedCandidate[]; note: string; run?: TraceEvent
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
      {!ready || layer === null ? <p data-testid="steps-note" className="mlp-note">{note}</p>
        : <StepsBody data={data} candidates={candidates} selected={selected} layer={layer} />}
    </section>
  )
}

function StepsBody({ data, candidates, selected, layer }: { data: NeuronData; candidates: VotedCandidate[]; selected: number; layer: number }) {
  const state = useNeuronLayer(layer)
  const mode = useNeuronMode((s) => s.mode)
  const [hover, setHover] = useState<Hover | null>(null)
  const dims = decodeSteps(data.steps!).mlpIn.length / (activationsOf(data).length / data.neurons)
  const s = useMemo(() => layerSteps(decodeSteps(data.steps!), activationsOf(data), layer, data.neurons, dims), [data, layer, dims])
  const weights = state.status === 'ready' && state.asset.manifest.neurons === data.neurons && state.asset.manifest.dims === dims ? state : null
  const rowB64 = data.candidates.find((c) => c.id === candidates[selected].id)?.row
  const marked = useMemo(() => {
    if (!weights || !s) return []
    const row = mode === 'vote' && rowB64 ? unpackFloat32(rowB64) : null
    return listedNeurons(weights.asset.manifest, weights.layer, layerActivations(activationsOf(data), layer, data.neurons), mode, row, weights.asset.g, data.invRms)
  }, [weights, s, mode, rowB64, data, layer])
  if (!s) return <p data-testid="steps-note" className="mlp-note">{MISMATCH}</p>
  const domain = gateDomain(s.gate)
  const named = [...marked].sort((a, b) => Math.abs(s.h[b]) - Math.abs(s.h[a])).slice(0, 3)
  const box = (chart: Hover['chart']) => hover?.chart === chart
    ? <HoverBox s={s} hover={hover} weights={weights} /> : null
  const pending = state.status === 'error'
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
        <GateUpScatter gate={s.gate} up={s.up} domain={domain} marked={marked} named={named} hover={hover} onHover={setHover} box={box('scatter')} />
      </Step>
      <Step n={3} name="SiLU" tex={String.raw`\operatorname{SiLU}(g) = g \cdot \sigma(g)`}
        text="The gate goes through SiLU. Same gate axis as above: each dot slides straight down onto the curve.">
        <SiluChart gate={s.gate} domain={domain} marked={marked} counts={gateCounts(s.gate)} hover={hover} onHover={setHover} box={box('silu')} />
      </Step>
      <Step n={4} name="Product" tex={String.raw`h = \operatorname{SiLU}(g) \odot u`}
        text="Each neuron’s up value is scaled by its gate. This is the neuron’s activation, the number “Inside one MLP” shows.">
        <ProductBars h={s.h} summary={productSummary(s.h)} marked={marked} hover={hover} onHover={setHover} box={box('product')} />
      </Step>
      <Step n={5} name="Down" tex={String.raw`\text{edit} = W_{\text{down}} \cdot h`}
        text={<>Each neuron adds its column of W<sub>down</sub>, scaled by its activation. The sum is what this layer’s MLP adds to the stream.</>}>
        {weights ? <DownStrips s={s} layer={weights.layer} dims={dims} marked={marked} /> : pending}
      </Step>
      <div className="embed-callouts">
        {MLP_STEP_CALLOUTS.map((c) => <Term key={c.label} id={c.id} section={c.section} variant="chip" testId="steps-callout">ⓘ {c.label}</Term>)}
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

function NormStrips({ s, w, eps }: { s: LayerSteps; w: Float32Array; eps: number }) {
  const { n, rms: r } = normalize(s.x, w, eps)
  const peakW = w.reduce((m, v) => Math.max(m, Math.abs(v)), 0) || 1
  return (
    <div>
      <StepStrip values={s.x} lim={2 * rms(s.x)} label="x, into the MLP" testId="strip-x" />
      <StepStrip values={w} lim={peakW} label={`÷ rms(x) = ${r.toFixed(1)}, ⊙ w`} testId="strip-w" />
      <StepStrip values={n} lim={2 * rms(n)} label="n, normalized" testId="strip-n" />
      <p className="mlp-scale-note">{s.x.length} numbers each; blue positive, red negative, colours clipped at 2× the typical size.</p>
    </div>
  )
}

function DownStrips({ s, layer, dims, marked }: { s: LayerSteps; layer: NeuronLayer; dims: number; marked: number[] }) {
  const top = [...marked].sort((a, b) => Math.abs(s.h[b]) * columnNorm(layer, dims, b) - Math.abs(s.h[a]) * columnNorm(layer, dims, a)).slice(0, 3)
  const lim = 2 * rms(s.edit)
  return (
    <div>
      {top.map((i) => <StepStrip key={i} values={contribution(layer, dims, i, s.h[i])} lim={lim} label={`N${i}: ${f2(s.h[i])} × column`} testId={`strip-c${i}`} labelClass="step-strip-mono" />)}
      <div className="step-strip-row"><span className="step-strip-label">+ {s.h.length - top.length} more</span><span className="mlp-note">…</span></div>
      <StepStrip values={s.edit} lim={lim} label="= edit" testId="strip-edit" labelClass="step-strip-sum" />
      <p data-testid="steps-ratio" className="mlp-scale-note">‖edit‖ = {Math.hypot(...s.edit).toFixed(0)},{' '}
        {editRatio(s.edit, s.x).toFixed(2)} of the stream’s length (the edit size card’s cell). Same colour scale on every row.</p>
    </div>
  )
}

// Placed next to the hovered mark, as in the mockup: to its left in the right half of the chart, to its right otherwise.
function HoverBox({ s, hover, weights }: { s: LayerSteps; hover: Hover; weights: { asset: NeuronAsset; layer: NeuronLayer } | null }) {
  const i = hover.neuron
  const left = hover.x > CHART_W / 2
  const style = { left: `${(hover.x / CHART_W) * 100}%`, top: `${(hover.y / hover.height) * 100}%`,
    transform: left ? 'translate(calc(-100% - 10px), 8px)' : 'translate(10px, 8px)' }
  const labels = weights ? neuronLabels(weights.asset.manifest, weights.layer, i, s.h[i] >= 0 ? 0 : 1).slice(0, 3).map((id) => visibleToken(weights.asset.text(id))) : []
  return (
    <div data-testid="steps-box" className="step-box" role="status" style={style}>
      <b>N{i}</b>
      <span>g {f2(s.gate[i])} → SiLU {f2(silu(s.gate[i]))}</span>
      <span>× u {f2(s.up[i])} = h {f2(s.h[i])}</span>
      {labels.length > 0 && <span className="step-box-labels">pushes toward {labels.join(', ')}…</span>}
    </div>
  )
}

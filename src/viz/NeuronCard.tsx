import { useMemo } from 'react'
import { NEURON_CALLOUTS } from '../glossary/callouts'
import { Term } from '../glossary/Term'
import type { NeuronAsset } from '../neurons/asset'
import type { NeuronLayer } from '../neurons/format'
import { behindTheVote, firingMost, layerActivations, reconcile, type VoteRow } from '../neurons/math'
import { useNeuronLayer } from '../neurons/useNeuronLayer'
import { unpackFloat32, unpackHalf } from '../trace/neuronPack'
import type { NeuronData, TraceEvent } from '../trace/types'
import { useCandidateSelection } from './candidateSelection'
import { useMlpLayerControl } from './mlpLayerSelection'
import { useNeuronMode, type NeuronMode as Mode } from './neuronMode'
import type { VotedCandidate } from './selectors'
import { visibleToken } from './tokenText'

const MODE_NAME: Record<Mode, string> = { vote: 'behind the vote', size: 'firing most' }
const LOADING = 'Loading the MLP weights (about 0.9 MB per layer, once per model version)…'
const MISMATCH = 'This run’s neuron data doesn’t match the MLP weights file.'
const NO_ROW = 'No neuron data for this candidate.'
const APPROX = 'The neuron votes don’t add up exactly to the votes card here. Runs saved with an earlier version of Tsumugi show this: its browser model rounded each MLP’s inputs.'
const BAR_MAX = 70

const signed = (v: number, digits = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}`
const BADGE_CLASS: Record<string, string> = { 'top 0.1%': 'strong', 'top 1%': 'mid', 'top 10%': 'low', 'not in top 10%': 'none' }

// Decoding 46 080 float16 values once per cycle's data, not per render.
const decoded = new WeakMap<NeuronData, Float32Array>()
const activationsOf = (d: NeuronData) => {
  let a = decoded.get(d)
  if (!a) { a = unpackHalf(d.activations); decoded.set(d, a) }
  return a
}

// "Inside one MLP": one layer's MLP vote for the selected candidate split into
// one vote per neuron (#16). Candidate selection is shared with the Logits and
// votes cards, the layer with both MLP cards; the mode is kept across remounts (neuronMode.ts).
export function NeuronCard({ data, candidates, note, run }: {
  data?: NeuronData; candidates?: VotedCandidate[]; note: string; run?: TraceEvent
}) {
  const ready = data && candidates && candidates.length > 0
  const { layer, layers, selected: sel, step } = useMlpLayerControl(ready ? candidates : undefined, run)
  return (
    <section data-testid="neuron-card" className="mlp-card">
      <h4 data-testid="neuron-title">Inside one MLP{layer !== null ? `: layer ${layer}` : ''}{layer !== null && (
        <span className="neuron-stepper">
          <button type="button" data-testid="neuron-prev" aria-label="Previous layer" disabled={layer <= 0} onClick={() => step(-1)}>◀</button>
          <button type="button" data-testid="neuron-next" aria-label="Next layer" disabled={layer >= layers - 1} onClick={() => step(1)}>▶</button>
        </span>
      )}{' '}<Term id="mlp-neuron" variant="icon" label="About MLP neurons" testId="neuron-info">ⓘ</Term></h4>
      {!ready || layer === null ? <p data-testid="neuron-note" className="mlp-note">{note}</p>
        : <NeuronBody data={data} candidates={candidates} selected={sel} layer={layer} />}
    </section>
  )
}

function NeuronBody({ data, candidates, selected, layer }: {
  data: NeuronData; candidates: VotedCandidate[]; selected: number; layer: number
}) {
  const mode = useNeuronMode((s) => s.mode)
  const setMode = useNeuronMode((s) => s.setMode)
  const select = useCandidateSelection((s) => s.select)
  const state = useNeuronLayer(layer)
  const cand = candidates[selected]
  return (
    <>
      <p className="mlp-caption">{mode === 'vote'
        ? 'The MLP’s edit is the sum of 1,536 neuron edits, so its vote splits into one vote per neuron. Click a layer in either card above to open it here.'
        : 'The neurons that changed the vector most for the last token, whatever they voted for. Size = |activation| × length of the neuron’s column.'}</p>
      <div className="mlp-controls">
        <div className="mlp-seg" role="group" aria-label="Which neurons">
          {(['vote', 'size'] as const).map((m) => (
            <button key={m} type="button" data-testid="neuron-mode-toggle" aria-pressed={m === mode}
              onClick={() => setMode(m)}>{MODE_NAME[m]}</button>
          ))}
        </div>
        {mode === 'vote' && (
          <div className="neuron-cands" role="group" aria-label="Candidate">candidate:{' '}
            {candidates.map((c, i) => (
              <button key={c.id} type="button" data-testid="neuron-cand" className="mlp-cand" aria-pressed={i === selected}
                onClick={() => select(candidates, i)}>{visibleToken(c)}</button>
            ))}
          </div>
        )}
      </div>
      {state.status === 'loading' || state.status === 'idle' ? <p data-testid="neuron-loading" className="mlp-note">{LOADING}</p>
        : state.status === 'error' ? (
          <p data-testid="neuron-error" className="mlp-note">Couldn’t load the MLP weights.{' '}
            <button type="button" data-testid="neuron-retry" onClick={state.retry}>Retry</button></p>
        ) : <NeuronTable data={data} asset={state.asset} weights={state.layer} layer={layer} mode={mode} cand={cand} />}
      <div className="embed-callouts">
        {NEURON_CALLOUTS.map((c) => (
          <Term key={c.label} id={c.id} section={c.section} tip={c.tip} variant="chip" testId="neuron-callout">ⓘ {c.label}</Term>
        ))}
      </div>
    </>
  )
}

function NeuronTable({ data, asset, weights, layer, mode, cand }: {
  data: NeuronData; asset: NeuronAsset; weights: NeuronLayer; layer: number; mode: Mode; cand: VotedCandidate
}) {
  const m = asset.manifest
  const fits = data.neurons === m.neurons && activationsOf(data).length === m.layers * m.neurons
  const rowB64 = data.candidates.find((c) => c.id === cand.id)?.row
  const acts = useMemo(() => (fits ? layerActivations(activationsOf(data), layer, m.neurons) : null), [fits, data, layer, m])
  const table = useMemo(() => {
    if (!acts || mode !== 'vote' || !rowB64) return null
    const row = unpackFloat32(rowB64)
    if (row.length !== m.dims) return null
    return reconcile(behindTheVote(m, weights, acts, row, asset.g, data.invRms), cand.votes.mlp[layer])
  }, [acts, mode, rowB64, m, weights, asset.g, data.invRms, cand, layer])
  const sizeRows = useMemo(() => (acts && mode === 'size' ? firingMost(m, weights, acts) : null), [acts, mode, m, weights])
  if (!acts) return <p data-testid="neuron-note" className="mlp-note">{MISMATCH}</p>
  const chips = (ids: number[]) => ids.map((id, j) => (
    <span key={j} data-testid="neuron-label" className="neuron-chip">{visibleToken(asset.text(id))}</span>
  ))
  if (mode === 'size' && sizeRows) {
    const rows = sizeRows
    const peak = Math.max(...rows.map((r) => r.size)) || 1
    return (
      <table data-testid="neuron-table" className="neuron-table">
        <thead><tr><th>neuron</th><th>activation</th><th>edit size</th><th>when it fires like this, it pushes toward</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.neuron} data-testid="neuron-row">
            <td className="tk">N{r.neuron}</td><td className="tk">{signed(r.activation, 1)}</td>
            <td><span className="neuron-size-bar" style={{ width: `${(r.size / peak) * BAR_MAX}px` }} />{r.size.toFixed(2)}</td>
            <td>{chips(r.labels)}</td>
          </tr>
        ))}</tbody>
      </table>
    )
  }
  if (!table) return <p data-testid="neuron-note" className="mlp-note">{NO_ROW}</p>
  const name = visibleToken(cand)
  const peak = Math.max(...table.rows.map((r) => Math.abs(r.vote))) || 1
  return (
    <>
      <table data-testid="neuron-table" className="neuron-table">
        <thead><tr><th>neuron</th><th>activation</th><th>vote for “{name}”</th>
          <th>when it fires like this, it pushes toward</th><th>“{name}” in that list</th></tr></thead>
        <tbody>
          {table.rows.map((r: VoteRow) => (
            <tr key={r.neuron} data-testid="neuron-row">
              <td className="tk">N{r.neuron}</td><td className="tk">{signed(r.activation, 1)}</td>
              <td><span className={`votes-bar ${r.vote >= 0 ? 'votes-bar-pos' : 'votes-bar-neg'}`}
                style={{ width: `${(Math.abs(r.vote) / peak) * BAR_MAX}px` }} /><span data-testid="neuron-vote">{signed(r.vote)}</span></td>
              <td>{r.away && <span data-testid="neuron-away" className="neuron-chip neuron-away">away from</span>}{chips(r.labels)}</td>
              <td><span data-testid="neuron-badge" className={`neuron-badge neuron-badge-${BADGE_CLASS[r.badge]}`}>
                {r.badge}{r.away ? ' (away)' : ''}</span></td>
            </tr>
          ))}
          <tr data-testid="neuron-rest" className="neuron-rest">
            <td colSpan={2}>the other {(m.neurons - table.rows.length).toLocaleString('en-US')} neurons</td>
            <td colSpan={3}>{signed(table.rest)}</td></tr>
          <tr data-testid="neuron-total" className="neuron-total">
            <td colSpan={2}>layer {layer} MLP, total</td>
            <td colSpan={3}>{table.exact ? signed(table.total) : `≈ ${signed(table.total)}`}{table.exact && (
              <span className="mlp-legend-note"> = the votes card’s cell</span>)}</td></tr>
        </tbody>
      </table>
      {!table.exact && <p data-testid="neuron-approx-note" className="mlp-scale-note">{APPROX}</p>}
    </>
  )
}

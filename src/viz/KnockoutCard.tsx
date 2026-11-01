import { useState, type ReactNode } from 'react'
import type { KnockoutLayer, KnockoutSweep, TokenInfo, TraceEvent } from '../trace/types'
import { Term } from '../glossary/Term'
import { knockoutEligibility, type Ineligible } from './knockoutEligibility'
import { knockoutRoles } from './knockoutRoles'
import { useKnockoutRequest } from './knockoutRequest'
import { voteFill } from './mlpMath'
import { useMlpLayer } from './mlpLayerSelection'
import { visibleToken } from './tokenText'

const LAYERS = 30
const MIN_SCALE = 0.01   // a flat sweep must not blow its noise up to full height

export const KNOCKOUT_PENDING_NOTE = 'Measure once the run has ended.'

type Run = Extract<TraceEvent, { type: 'run-start' }>
type Running = { recordId: string; cycle: number; baseline: number | null; layers: KnockoutLayer[] } | null

function noteFor(reason: Ineligible, run: Run): { text: string; action: boolean } {
  switch (reason) {
    case 'sim': return { text: 'Real model only: simulated runs have no MLP to measure.', action: true }
    case 'no-model': return { text: 'Load the real model to measure knockouts.', action: true }
    case 'old-run': return { text: 'Recorded with an earlier version of the model — generate the run again to measure knockouts.', action: false }
    case 'knockout-run': return { text: `This run already has MLP ${run.knockout?.layer} removed — open the original run to measure.`, action: false }
    case 'too-long': return { text: 'This run is longer than 512 tokens, too long to measure.', action: false }
  }
}

const points = (d: number) => { const n = Math.round(d * 100); return `${n < 0 ? '−' : '+'}${Math.abs(n)} pts` }

// "MLP 15 · preparatory · −58 pts"; a decisive layer names the token change instead
function chipText(r: { layer: number; role: string }, e: KnockoutLayer | undefined, baseline: number | null, chosen: TokenInfo): string {
  const head = `MLP ${r.layer} · ${r.role}`
  if (!e || baseline === null) return head
  return r.role === 'decisive' ? `${head} · ${quoted(chosen)} → ${quoted(e.top)}` : `${head} · ${points(e.prob - baseline)}`
}

const pct = (p: number) => `${Math.round(p * 100)}%`
const quoted = (t: TokenInfo) => `“${visibleToken(t)}”`

// "Which layers the chosen token depends on": removes one MLP at a time and
// shows how the chosen token's probability changes (layout after mockup A).
export function KnockoutCard({ run, cycle, seqLen, chosen, votes, sweep, running, model, recordId, noteAction }: {
  run: Run
  cycle: number
  seqLen: number
  chosen?: TokenInfo           // absent while the cycle's sample is not in the trace yet
  votes?: number[]             // the chosen token's direct MLP vote per layer
  sweep?: KnockoutSweep
  running: Running
  model: { knockout: boolean; revision: string } | null
  recordId?: string            // absent while the run has no record yet
  noteAction?: ReactNode
}) {
  const measure = useKnockoutRequest((s) => s.measure)
  const runKnockout = useKnockoutRequest((s) => s.runKnockout)
  const failure = useKnockoutRequest((s) => s.error)
  const select = useMlpLayer((s) => s.select)
  const [picked, setPicked] = useState<number | null>(null)
  const title = <h4>Which layers the chosen token depends on{' '}
    <Term id="knockout" variant="icon" label="About knockouts" testId="knockout-info">ⓘ</Term></h4>

  const eligible = knockoutEligibility(run, seqLen, model)
  if (!eligible.ok) {
    const n = noteFor(eligible.reason, run)
    return <section data-testid="knockout-card" className="mlp-card">{title}
      <p data-testid="knockout-note" className="mlp-note">{n.text} {n.action && noteAction}</p></section>
  }
  if (!chosen || !recordId) {
    return <section data-testid="knockout-card" className="mlp-card">{title}
      <p data-testid="knockout-note" className="mlp-note">{KNOCKOUT_PENDING_NOTE}</p></section>
  }

  const runningHere = running !== null && running.recordId === recordId && running.cycle === cycle
  const shown = runningHere ? { baseline: running.baseline, layers: running.layers }
    : sweep ? { baseline: sweep.baseline, layers: sweep.layers } : null
  const byLayer = new Map((shown?.layers ?? []).map((l) => [l.layer, l]))
  const baseline = shown?.baseline ?? null
  const drops = baseline === null ? [] : [...byLayer.values()].map((l) => baseline - l.prob)
  const scale = Math.max(MIN_SCALE, ...drops.map(Math.abs))
  const votePeak = Math.max(0, ...(votes ?? []).map(Math.abs)) || 1
  const roles = sweep && !runningHere ? knockoutRoles(sweep, votes ?? []) : []
  const busy = running !== null
  const pick = (l: number) => { setPicked(l); select(run, l) }
  const entry = picked === null ? undefined : byLayer.get(picked)
  const columns = { gridTemplateColumns: `repeat(${LAYERS}, 1fr)` }

  return (
    <section data-testid="knockout-card" className="mlp-card">
      {title}
      <p className="mlp-caption">Removes one layer’s MLP at a time and asks the model again for this cycle’s next
        token: {LAYERS + 1} passes over the whole text, about two seconds for a short text on a graphics card.</p>
      {failure && failure.recordId === recordId && failure.cycle === cycle && (
        <p data-testid="knockout-error" className="mlp-note">Measuring failed: {failure.message}</p>
      )}
      {(!sweep || runningHere) && (
        <div><button type="button" className="knockout-measure" disabled={busy || !measure}
          onClick={() => measure?.(cycle)}>Measure all 30 layers</button></div>
      )}
      {sweep && !runningHere && <p className="mlp-scale-note">done for this cycle</p>}
      {runningHere && <p className="mlp-scale-note" aria-live="polite">Measuring… {byLayer.size} of {LAYERS} layers</p>}
      <div className="knockout-grid">
        <div className="knockout-row knockout-bars-row">
          <span className="knockout-label">{quoted(chosen)} without this layer</span>
          <div className="knockout-cells" style={columns}>
            {Array.from({ length: LAYERS }, (_, l) => {
              const e = byLayer.get(l)
              if (!e || baseline === null) return <div key={l} className="knockout-col" />
              const drop = baseline - e.prob
              return (
                <div key={l} className="knockout-col">
                  <button type="button" data-testid="knockout-bar" data-layer={l} data-drop={drop.toFixed(2)}
                    data-flips={String(e.flips)} aria-pressed={picked === l}
                    aria-label={`MLP ${l}: ${quoted(chosen)} ${pct(baseline)} → ${pct(e.prob)}`}
                    className={`knockout-bar ${drop >= 0 ? 'knockout-bar-drop' : 'knockout-bar-rise'}${e.flips ? ' knockout-bar-flips' : ''}`}
                    style={{ height: `${Math.max(2, (Math.abs(drop) / scale) * 100)}%` }}
                    onClick={() => pick(l)} />
                </div>
              )
            })}
          </div>
        </div>
        {votes && (
          <div className="knockout-row">
            <span className="knockout-label">direct vote</span>
            <div className="knockout-cells knockout-votes" style={columns}>
              {Array.from({ length: LAYERS }, (_, l) => (
                <div key={l} data-testid="knockout-vote" data-layer={l} data-value={(votes[l] ?? 0).toFixed(2)}
                  style={{ background: voteFill(votes[l] ?? 0, votePeak) }} />
              ))}
            </div>
          </div>
        )}
        <div className="knockout-row">
          <span />
          <div className="knockout-cells mlp-axis" style={columns}>
            {Array.from({ length: LAYERS }, (_, l) => <span key={l}>{l % 5 === 0 || l === LAYERS - 1 ? l : ''}</span>)}
          </div>
        </div>
      </div>
      {roles.length > 0 && (
        <div className="knockout-chips">
          {roles.map((r) => (
            <button key={r.layer} type="button" data-testid="knockout-chip" data-role={r.role}
              className={`knockout-chip knockout-chip-${r.role}`} onClick={() => pick(r.layer)}>
              {chipText(r, byLayer.get(r.layer), baseline, chosen)}</button>
          ))}
        </div>
      )}
      {entry && baseline !== null && picked !== null && (
        <div className="knockout-tip">
          <span data-testid="knockout-detail">MLP {picked} — {quoted(chosen)} {pct(baseline)} → {pct(entry.prob)},{' '}
            {entry.top.id !== chosen.id ? `top token becomes ${quoted(entry.top)}` : 'top token unchanged'}</span>{' '}
          <button type="button" className="knockout-run" disabled={busy || !runKnockout}
            onClick={() => runKnockout?.(picked, cycle)}>Run with MLP {picked} off and compare</button>
        </div>
      )}
    </section>
  )
}

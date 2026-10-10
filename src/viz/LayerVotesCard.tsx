import { useState } from 'react'
import { Term } from '../glossary/Term'
import type { LayerVotes } from '../trace/types'
import { selectedCandidate, useCandidateSelection } from './candidateSelection'
import { Formula } from './Formula'
import { blockTotals, layerTotals, topMlpLayers, voteFill } from './mlpMath'
import type { VotedCandidate } from './selectors'
import { visibleToken } from './tokenText'

type Blocks = 'mlp' | 'attn' | 'both'
const BLOCKS_NAME: Record<Blocks, string> = { mlp: 'MLP', attn: 'attention', both: 'both' }

const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`
const sum = (a: number[]) => a.reduce((s, v) => s + v, 0)

// The per-layer votes the grid shows for the chosen blocks.
function shown(v: LayerVotes, blocks: Blocks): number[] {
  if (blocks === 'mlp') return v.mlp
  if (blocks === 'attn') return v.attn
  return v.mlp.map((m, l) => m + v.attn[l])
}

function cellText(blocks: Blocks, layer: number, token: string, v: number): string {
  const who = blocks === 'both' ? `layer ${layer} (attention + MLP)` : `${BLOCKS_NAME[blocks]} ${layer}`
  return `${who} → “${visibleToken(token)}”: ${signed(v)} to the score`
}

// "What each layer voted for": the score split exactly into one share per
// block (final norm frozen at its real scale). The candidate selection is
// shared with the Logits card; the blocks shown and the hovered cell are
// local view state.
export function LayerVotesCard({ candidates, note }: { candidates?: VotedCandidate[]; note: string }) {
  const pick = useCandidateSelection((s) => s.pick)
  const select = useCandidateSelection((s) => s.select)
  return (
    <section data-testid="votes-card" className="mlp-card">
      <h4>What each layer voted for{' '}
        <Term id="votes" variant="icon" label="About per-layer votes" testId="votes-info">ⓘ</Term></h4>
      {!candidates || candidates.length === 0 ? <p data-testid="votes-note" className="mlp-note">{note}</p> : (
        <VotesBody candidates={candidates} selected={selectedCandidate(pick, candidates)}
          onSelect={(i) => select(candidates, i)} />
      )}
    </section>
  )
}

function VotesBody({ candidates, selected, onSelect }: {
  candidates: VotedCandidate[]; selected: number; onSelect: (i: number) => void
}) {
  const [blocks, setBlocks] = useState<Blocks>('mlp')
  const [hover, setHover] = useState<string | null>(null)
  const layers = candidates[0].votes.mlp.length
  const values = candidates.map((c) => shown(c.votes, blocks))
  const totals = values.map(sum)
  const peak = Math.max(...values.flat().map(Math.abs)) || 1
  const totalPeak = Math.max(...totals.map(Math.abs)) || 1
  const columns = { gridTemplateColumns: `repeat(${layers}, 1fr)` }
  return (
    <>
      <p className="mlp-caption">The <Term id="logit">score</Term> on the Logits card splits exactly into one
        share per block. Each cell is how much one layer added to (blue) or took from (red) a candidate’s
        score.</p>
      <Formula testId="votes-formula" className="card-formula" inline tex={String.raw`\displaystyle \begin{aligned} &\text{score}(c) = e \cdot E_c + \sum_{\ell} a_\ell \cdot E_c + \sum_{\ell} m_\ell \cdot E_c \\ &\text{each part scaled by the final norm} \end{aligned}`} />
      <div className="mlp-controls">
        <div className="mlp-seg" role="group" aria-label="Which blocks">
          {(['mlp', 'attn', 'both'] as const).map((b) => (
            <button key={b} type="button" data-testid="votes-block-toggle" aria-pressed={b === blocks}
              onClick={() => setBlocks(b)}>{BLOCKS_NAME[b]}</button>
          ))}
        </div>
        <div data-testid="votes-legend" className="mlp-legend">−{peak.toFixed(1)}{' '}
          <span className="mlp-ramp-diverging" aria-hidden="true" /> +{peak.toFixed(1)}
          <span className="mlp-legend-note">square-root scale, so small votes stay visible</span></div>
      </div>
      <div className="votes-grid" role="group" aria-label={`${BLOCKS_NAME[blocks]} votes, candidates by layers`}
        onMouseLeave={() => setHover(null)}>
        {candidates.map((c, r) => (
          <div key={c.id} className="votes-grid-row">
            <button type="button" data-testid="vote-row-label" aria-pressed={r === selected}
              className="mlp-cand" onClick={() => onSelect(r)}>{visibleToken(c.text)}</button>
            <div className="mlp-cells votes-cells" style={columns}>
              {values[r].map((v, l) => (
                <div key={l} data-testid="vote-cell" data-row={r} data-layer={l} data-value={v.toFixed(2)}
                  style={{ background: voteFill(v, peak) }} onClick={() => onSelect(r)}
                  onMouseEnter={() => setHover(cellText(blocks, l, c.text, v))} />
              ))}
            </div>
            <span data-testid="vote-row-total" className="votes-total-cell">
              <span className="votes-swatch" style={{ background: voteFill(totals[r], totalPeak) }} aria-hidden="true" />
              Σ {signed(totals[r])}</span>
          </div>
        ))}
        <div className="votes-grid-row">
          <span />
          <div className="mlp-cells mlp-axis" style={columns}>
            {Array.from({ length: layers }, (_, l) => <span key={l}>{l % 5 === 0 ? `L${l}` : ''}</span>)}
          </div>
          <span />
        </div>
      </div>
      <p data-testid="vote-readout" className="votes-readout" aria-live="polite">{hover ?? 'Hover a cell to see one layer’s vote.'}</p>
      <div className="votes-split">
        <RunningScore candidates={candidates} selected={selected} />
        <Breakdown candidate={candidates[selected]} layers={layers} />
      </div>
    </>
  )
}

// The running-score chart's frame, in viewBox units.
const W = 440, H = 190, X0 = 34, X1 = 380, Y0 = 160, TOP = 12

function RunningScore({ candidates, selected }: { candidates: VotedCandidate[]; selected: number }) {
  const series = candidates.map((c) => layerTotals(c.votes))
  const all = series.flat()
  const lo = Math.min(0, ...all), hi = Math.max(...all)
  const steps = series[0].length - 1   // the embedding, then one point per layer
  const x = (j: number) => X0 + (j / steps) * (X1 - X0)
  const y = (v: number) => Y0 - ((v - lo) / (hi - lo || 1)) * (Y0 - TOP)
  const step = Math.max(1, Math.ceil((hi - lo) / 6))
  const ticks: number[] = []
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t)
  const layers = steps
  const layerTicks = [...new Set([0, 10, 20, layers - 1].filter((l) => l < layers))]
  // names at the line ends for the selected candidate and the four highest
  // final scores (ten names would pile up), pushed apart so they don't overlap
  const named = new Set([selected, ...series.map((s, r) => ({ r, end: s.at(-1)! }))
    .sort((a, b) => b.end - a.end).slice(0, 4).map((e) => e.r)])
  const ends = series.map((s, r) => ({ r, y: y(s.at(-1)!) })).filter((e) => named.has(e.r)).sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 10)
  const sel = candidates[selected]
  return (
    <figure className="mlp-chart-fig">
      <figcaption className="mlp-chart-label">running score, “{visibleToken(sel.text)}” against the rest — ends
        at <span data-testid="vote-final">{sel.logit.toFixed(2)}</span></figcaption>
      <svg data-testid="votes-lens" className="mlp-chart" viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label="each candidate's score building up across the layers">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X0} x2={X1} y1={y(t)} y2={y(t)} className={t === 0 ? 'mlp-ref-line' : 'mlp-grid-line'} />
            <text x={X0 - 5} y={y(t) + 3} textAnchor="end" className="mlp-tick">{t}</text>
          </g>
        ))}
        {layerTicks.map((l) => (
          <text key={l} x={x(l + 1)} y={Y0 + 14} textAnchor="middle" className="mlp-tick">L{l}</text>
        ))}
        {series.map((s, r) => r !== selected && (
          <polyline key={candidates[r].id} data-testid="vote-line" data-selected="false" data-final={s.at(-1)!.toFixed(2)}
            className="vote-line" points={s.map((v, j) => `${x(j).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} />
        ))}
        <polyline data-testid="vote-line" data-selected="true" data-final={sel.logit.toFixed(2)} className="vote-line-selected"
          points={series[selected].map((v, j) => `${x(j).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} />
        <circle cx={x(steps)} cy={y(series[selected].at(-1)!)} r={3.5} className="vote-line-end" />
        {ends.map(({ r, y: ly }) => (
          <text key={candidates[r].id} data-testid="vote-line-label" x={X1 + 6} y={ly + 3}
            className={r === selected ? 'mlp-tick vote-label-selected' : 'mlp-tick'}>{visibleToken(candidates[r].text)}</text>
        ))}
      </svg>
      <p className="mlp-scale-note">Each layer adds its attention and MLP votes; the selected candidate overtakes
        its rivals where its line crosses theirs.</p>
    </figure>
  )
}

const BAR_MAX = 90

function Breakdown({ candidate, layers }: { candidate: VotedCandidate; layers: number }) {
  const totals = blockTotals(candidate.votes)
  const rows: Array<{ label: string; value: number; sub?: boolean }> = [
    { label: 'embedding of the last token', value: totals.embed },
    { label: `all ${layers} attention blocks`, value: totals.attn },
    { label: `all ${layers} MLP blocks`, value: totals.mlp },
    ...topMlpLayers(candidate.votes).map(({ layer, value }) => ({ label: `of which MLP ${layer}`, value, sub: true })),
  ]
  const barMax = Math.max(...rows.map((r) => Math.abs(r.value))) || 1
  return (
    <table className="votes-breakdown">
      <caption>where the score of “{visibleToken(candidate.text)}” comes from</caption>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label} data-testid="votes-row" className={r.sub ? 'votes-sub' : undefined}>
            <th scope="row">{r.label}</th>
            <td><span className={`votes-bar ${r.value >= 0 ? 'votes-bar-pos' : 'votes-bar-neg'}`}
              style={{ width: `${(Math.abs(r.value) / barMax) * BAR_MAX}px` }} /></td>
            <td>{signed(r.value)}</td>
          </tr>
        ))}
        <tr data-testid="votes-total" className="votes-total">
          <th scope="row">score on the Logits card</th><td /><td>{candidate.logit.toFixed(2)}</td>
        </tr>
      </tbody>
    </table>
  )
}

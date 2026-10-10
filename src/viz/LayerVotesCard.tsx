import { Term } from '../glossary/Term'
import { selectedCandidate, useCandidateSelection } from './candidateSelection'
import { Formula } from './Formula'
import { blockTotals, runningTotal, topMlpLayers, votePeak } from './mlpMath'
import type { VotedCandidate } from './selectors'
import { cellFill } from './stripMath'
import { visibleToken } from './tokenText'

const CELL = 8, ROW_H = 14, LABEL_W = 64, LINE_W = 260, LINE_H = 90, BAR_MAX = 90

const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`

// "What each layer voted for": the score split exactly into one share per
// block (final norm frozen at its real scale). The candidate selection is
// shared with the Logits card.
export function LayerVotesCard({ candidates, cycle, note }: { candidates?: VotedCandidate[]; cycle: number; note: string }) {
  const pick = useCandidateSelection((s) => s.pick)
  const select = useCandidateSelection((s) => s.select)
  return (
    <section data-testid="votes-card" className="mlp-card">
      <h4>What each layer voted for{' '}
        <Term id="votes" variant="icon" label="About per-layer votes" testId="votes-info">ⓘ</Term></h4>
      {!candidates || candidates.length === 0 ? <p data-testid="votes-note" className="mlp-note">{note}</p> : (
        <VotesBody candidates={candidates} selected={selectedCandidate(pick, cycle, candidates.length)}
          onSelect={(i) => select(cycle, i)} />
      )}
    </section>
  )
}

function VotesBody({ candidates, selected, onSelect }: {
  candidates: VotedCandidate[]; selected: number; onSelect: (i: number) => void
}) {
  const layers = candidates[0].votes.mlp.length
  const peak = votePeak(candidates)
  const sel = candidates[selected]
  const totals = blockTotals(sel.votes)
  const rows: Array<{ label: string; value: number; sub?: boolean }> = [
    { label: 'embedding of the last token', value: totals.embed },
    { label: `all ${layers} attention blocks`, value: totals.attn },
    { label: `all ${layers} MLP blocks`, value: totals.mlp },
    ...topMlpLayers(sel.votes).map(({ layer, value }) => ({ label: `of which MLP ${layer}`, value, sub: true })),
  ]
  const barMax = Math.max(...rows.map((r) => Math.abs(r.value))) || 1
  const w = LABEL_W + layers * CELL, h = candidates.length * ROW_H
  return (
    <>
      <p className="mlp-caption">The <Term id="logit">score</Term> on the Logits card splits exactly into one
        share per block. Each cell is how much one layer’s MLP added to (blue) or took from (red) a candidate’s
        score.</p>
      <Formula testId="votes-formula" className="card-formula" inline tex={String.raw`\displaystyle \begin{aligned} &\text{score}(c) = e \cdot E_c + \sum_{\ell} a_\ell \cdot E_c + \sum_{\ell} m_\ell \cdot E_c \\ &\text{each part scaled by the final norm} \end{aligned}`} />
      <figure className="edit-map-fig">
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="group" aria-label="MLP votes, candidates by layers">
          {candidates.map((c, r) => (
            <g key={c.id}>
              <text data-testid="vote-row-label" data-row={r} data-selected={String(r === selected)}
                className={`edit-row-label${r === selected ? ' edit-row-selected' : ''}`}
                x={LABEL_W - 4} y={r * ROW_H + 10} textAnchor="end" role="button" tabIndex={0}
                onClick={() => onSelect(r)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(r) }}>
                {visibleToken(c.text)}
              </text>
              {c.votes.mlp.map((v, l) => (
                <rect key={l} data-testid="vote-cell" data-row={r} data-layer={l} x={LABEL_W + l * CELL} y={r * ROW_H}
                  width={CELL - 1} height={ROW_H - 1} fill={cellFill(v, peak)} onClick={() => onSelect(r)}>
                  <title>{`MLP ${l} → “${visibleToken(c.text)}”: ${signed(v)} to the score`}</title>
                </rect>
              ))}
            </g>
          ))}
        </svg>
      </figure>
      <RunningTotal candidates={candidates} selected={selected} />
      <table className="votes-breakdown">
        <caption>where the score of “{visibleToken(sel.text)}” comes from</caption>
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
            <th scope="row">score on the Logits card</th><td /><td>{sel.logit.toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
    </>
  )
}

function RunningTotal({ candidates, selected }: { candidates: VotedCandidate[]; selected: number }) {
  const series = candidates.map((c) => runningTotal(c.votes))
  const all = series.flat()
  const lo = Math.min(0, ...all), hi = Math.max(...all)
  const steps = series[0].length - 1
  const x = (j: number) => (j / steps) * (LINE_W - 4) + 2
  const y = (v: number) => LINE_H - 4 - ((v - lo) / (hi - lo || 1)) * (LINE_H - 8)
  const sel = candidates[selected]
  return (
    <figure className="votes-lens">
      <figcaption>running score, “{visibleToken(sel.text)}” against the rest — ends at{' '}
        <span data-testid="vote-final">{sel.logit.toFixed(2)}</span></figcaption>
      <svg width={LINE_W} height={LINE_H} viewBox={`0 0 ${LINE_W} ${LINE_H}`} role="img"
        aria-label="each candidate's score building up across the layers">
        {series.map((s, r) => (
          <polyline key={candidates[r].id} data-testid="vote-line" data-row={r} data-selected={String(r === selected)}
            data-final={s.at(-1)!.toFixed(2)} className={r === selected ? 'vote-line-selected' : 'vote-line'}
            points={s.map((v, j) => `${x(j).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} />
        ))}
      </svg>
    </figure>
  )
}

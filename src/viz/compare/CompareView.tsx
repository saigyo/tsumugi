import { useState } from 'react'
import type { RunRecord } from '../../app/runsStore'
import type { AttentionHead, TokenInfo, TraceEvent } from '../../trace/types'
import { AttentionHeatmap } from '../AttentionHeatmap'
import { Thumb } from '../AttentionGridExplorer'
import { chatLayout, distributionFor, latestOfType, visibleTokens } from '../selectors'
import { alignRuns, combinedDistribution, pairedHeads, type CombinedRow, type PairedHead } from './compareSelectors'
import { visibleToken } from '../tokenText'
import { percent, points } from '../../ui/format'

const runTokens = (events: TraceEvent[]): TokenInfo[] => {
  const { prompt, generated } = visibleTokens(events, events.length - 1)
  return [...prompt, ...generated]
}

const gridThumbFor = (events: TraceEvent[], layer: number, head: number): number[][] | null => {
  const grid = latestOfType(events, events.length - 1, 'attention-grid')
  return grid?.cells.find((c) => c.layer === layer && c.head === head)?.thumb ?? null
}

function MetaRow({ label, a, b }: { label: string; a: string; b: string }) {
  return (
    <div data-testid="cmp-meta-row" data-diff={String(a !== b)} className="cmp-meta-row">
      <span className="cmp-meta-label">{label}</span>
      <span className="cmp-meta-value">{a}</span>
      <span className="cmp-meta-value">{b}</span>
    </div>
  )
}

function DistSide({ events, cycle, name, lastCycle }: {
  events: TraceEvent[]; cycle: number; name: string; lastCycle: number
}) {
  const d = distributionFor(events, cycle)
  if (!d) return <div data-testid="cmp-ended" className="cmp-ended">{name}: run ended at cycle {lastCycle}</div>
  return (
    <div data-testid="cmp-dist-side" className="cmp-dist-side">
      <span className="cmp-side-name">{name} · T={d.softmax.temperature}</span>
      {d.softmax.topK.map((t) => (
        <div key={t.id} data-testid="cmp-bar-row" data-chosen={String(t.id === d.sample.chosen.id)}
          className="cmp-bar-row">
          <span className="cmp-bar-token"><span className="tok tok-sm" title={visibleToken(t)}><span className="chip-text">{visibleToken(t)}</span></span></span>
          <span className={`bar${t.id === d.sample.chosen.id ? ' bar-chosen' : ''}`} style={{ width: `${Math.max(2, Math.round(t.prob * 100))}%` }} />
          <span className="cmp-bar-prob">{percent(t.prob)}</span>
        </div>
      ))}
    </div>
  )
}

type DeltaSort = 'delta' | 'a' | 'b'

function CombinedDist({ rows, chosenA, chosenB, sort, nameA, nameB }: {
  rows: CombinedRow[]; chosenA: number; chosenB: number; sort: DeltaSort
  nameA: string; nameB: string
}) {
  const sorted = [...rows].sort((x, y) =>
    sort === 'a' ? (y.pA ?? 0) - (x.pA ?? 0)
      : sort === 'b' ? (y.pB ?? 0) - (x.pB ?? 0)
        : Math.abs(y.delta) - Math.abs(x.delta))
  return (
    <div className="cmp-combined">
      <div className="cmp-delta-row cmp-delta-head" aria-hidden="true">
        <span /><span className="cmp-side-name">{nameA}</span><span /><span>Δ</span>
        <span className="cmp-side-name">{nameB}</span>
      </div>
      {sorted.map((r) => (
        <div key={r.id} data-testid="cmp-delta-row" data-token={r.text} data-approx={String(r.approx)}
          data-chosen-a={String(r.id === chosenA)} data-chosen-b={String(r.id === chosenB)}
          className="cmp-delta-row">
          <span className="cmp-bar-token"><span className="tok tok-sm" title={visibleToken(r)}><span className="chip-text">{visibleToken(r)}</span></span></span>
          <span className="cmp-delta-p" data-chosen={String(r.id === chosenA)}>{percent(r.pA)}</span>
          <span className="cmp-delta-bar">
            <span className="cmp-delta-fill" data-sign={r.delta > 0 ? 'a' : r.delta < 0 ? 'b' : 'zero'}
              style={{ width: `${Math.abs(r.delta) * 50}%` }} />
          </span>
          <span className="cmp-delta-value">{points(r.delta, r.approx)}</span>
          <span className="cmp-delta-p" data-chosen={String(r.id === chosenB)}>{percent(r.pB)}</span>
        </div>
      ))}
      {rows.some((r) => r.approx) && (
        <p data-testid="cmp-approx-note" className="fine attn-note">Tokens outside a run's stored top-k
          count as 0%, so those deltas are lower bounds.</p>
      )}
    </div>
  )
}

function AttnSide({ events, head, pair, name, cycle, lastCycle }: {
  events: TraceEvent[]; head?: AttentionHead; pair: PairedHead; name: string
  cycle: number; lastCycle: number
}) {
  if (cycle > lastCycle)
    return <div data-testid="cmp-ended" className="cmp-ended">{name}: run ended at cycle {lastCycle}</div>
  if (head) {
    return (
      <div data-testid="cmp-attn-side" className="cmp-attn-side">
        <span className="cmp-side-name">{name}</span>
        <AttentionHeatmap heads={[head]} tokens={runTokens(events)} chat={chatLayout(events)} foldToggle={false} />
      </div>
    )
  }
  const thumb = gridThumbFor(events, pair.layer, pair.head)
  return (
    <div data-testid="cmp-attn-side" className="cmp-attn-side">
      <span className="cmp-side-name">{name}</span>
      {thumb ? (
        <div data-testid="cmp-fallback" className="cmp-fallback">
          <Thumb thumb={thumb} />
          <p className="note">Not among this run's detected heads at this cycle; run-level thumbnail shown
            (full matrices are kept only for each cycle's showcase heads).</p>
        </div>
      ) : (
        <p data-testid="cmp-fallback" className="cmp-fallback note">Not among this run's detected heads
          at this cycle; no full matrix captured.</p>
      )}
    </div>
  )
}

export function CompareView({ a, b, initialCycle }: { a: RunRecord; b: RunRecord; initialCycle?: number }) {
  const aligned = alignRuns(a.events, b.events)
  const [cycle, setCycle] = useState<number | null>(initialCycle ?? null)
  const [headKey, setHeadKey] = useState<string | null>(null)
  const [distMode, setDistMode] = useState<'side' | 'combined'>('side')
  const [deltaSort, setDeltaSort] = useState<DeltaSort>('delta')
  const heads = cycle !== null ? pairedHeads(a.events, b.events, cycle) : []
  const selectedPair = heads.find((h) => `${h.layer}-${h.head}` === headKey) ?? heads[0]
  // every panel carries the run identity, matching the stream labels
  const nameA = `A #${a.meta.seq}`
  const nameB = `B #${b.meta.seq}`
  const selectCycle = (c: number) => { setCycle(c); setHeadKey(null) }

  const stream = (which: 'a' | 'b') => {
    const record = which === 'a' ? a : b
    const prompt = which === 'a' ? aligned.promptA : aligned.promptB
    const chosen = which === 'a' ? aligned.chosenA : aligned.chosenB
    return (
      <div data-testid={`cmp-stream-${which}`} className="cmp-stream">
        <span className="cmp-side-name">{which.toUpperCase()} #{record.meta.seq}</span>
        {prompt.map((t, i) => <span key={`p${i}`} className="cmp-token token token-prompt">{visibleToken(t)}</span>)}
        <span className="cmp-stream-divider" aria-hidden="true" />
        {chosen.map((t, c) => (
          <button key={`c${c}`} data-testid="cmp-token" data-fork={String(aligned.forkCycle === c)}
            data-selected={String(cycle === c)} className={`cmp-token token token-generated${cycle === c ? ' is-picked' : ''}`}
            title={`inspect cycle ${c}`} onClick={() => selectCycle(c)}>{visibleToken(t)}</button>
        ))}
      </div>
    )
  }

  return (
    <div data-testid="compare-view" className="compare-view">
      <div className="cmp-header">
        <div className="cmp-meta-row cmp-meta-head" aria-hidden="true">
          <span className="cmp-meta-label" />
          <span className="cmp-side-name">{nameA}</span>
          <span className="cmp-side-name">{nameB}</span>
        </div>
        <MetaRow label="Prompt" a={a.meta.prompt} b={b.meta.prompt} />
        <MetaRow label="Temperature" a={String(a.meta.params.temperature)} b={String(b.meta.params.temperature)} />
        <MetaRow label="Top-k" a={String(a.meta.params.topK)} b={String(b.meta.params.topK)} />
        <MetaRow label="Max tokens" a={String(a.meta.params.maxNewTokens)} b={String(b.meta.params.maxNewTokens)} />
        <MetaRow label="Mode" a={a.meta.mode} b={b.meta.mode} />
        <MetaRow label="Model" a={a.meta.modelId ?? '—'} b={b.meta.modelId ?? '—'} />
        <MetaRow label="Ended" a={a.meta.reason} b={b.meta.reason} />
      </div>
      {!aligned.samePrompt && (
        <p data-testid="cmp-badge" className="note">The prompts differ, so the runs are lined up by generation cycle.</p>
      )}
      {aligned.samePrompt && aligned.forkCycle === null
        && aligned.chosenA.length === aligned.chosenB.length && (
        <p data-testid="cmp-note" className="note">Identical outputs.</p>
      )}
      {stream('a')}
      {stream('b')}
      <div className="cmp-ruler">
        {Array.from({ length: aligned.maxCycles }, (_, c) => (
          <button key={c} data-testid="cmp-tick" data-selected={String(cycle === c)}
            data-fork={String(aligned.forkCycle === c)} className={`btn btn-sm cmp-tick${cycle === c ? ' is-view' : ''}`}
            onClick={() => selectCycle(c)}>{c}</button>
        ))}
      </div>
      {cycle !== null && (
        <>
          <h3>Cycle {cycle} · Distributions</h3>
          {(() => {
            const combined = distMode === 'combined' ? combinedDistribution(a.events, b.events, cycle) : undefined
            // prefixes are identical up to and including the fork cycle's input;
            // past it (or with different prompts) each distribution conditions
            // on its own text and token deltas stop being apples-to-apples
            const diverged = !aligned.samePrompt
              || (aligned.forkCycle !== null && cycle > aligned.forkCycle)
            return (
              <>
                <div className="cmp-dist-controls">
                  <div className="seg" role="group" aria-label="Distribution view">
                    <button data-testid="dist-mode-side" className={`seg-opt${distMode === 'side' ? ' is-view' : ''}`}
                      data-active={String(distMode === 'side')} aria-pressed={distMode === 'side'} onClick={() => setDistMode('side')}>Side by side</button>
                    <button data-testid="dist-mode-combined" className={`seg-opt${distMode === 'combined' ? ' is-view' : ''}`}
                      data-active={String(distMode === 'combined')} aria-pressed={distMode === 'combined'} onClick={() => setDistMode('combined')}>Combined</button>
                  </div>
                  {combined && (
                    <label className="grid-sort-label">sort{' '}
                      <select data-testid="dist-sort" className="btn" value={deltaSort}
                        onChange={(e) => setDeltaSort(e.target.value as DeltaSort)}>
                        <option value="delta">|Δ|</option>
                        <option value="a">A</option>
                        <option value="b">B</option>
                      </select>
                    </label>
                  )}
                </div>
                {combined && diverged && (
                  <p data-testid="cmp-divergence-note" className="note">
                    {aligned.samePrompt
                      ? `The runs' prefixes differ from cycle ${(aligned.forkCycle ?? 0) + 1} on, so each distribution conditions on its own text.`
                      : 'The prompts differ, so the distributions condition on different text.'}
                  </p>
                )}
                {combined ? (
                  <CombinedDist rows={combined.rows} chosenA={combined.chosenA} chosenB={combined.chosenB}
                    sort={deltaSort} nameA={nameA} nameB={nameB} />
                ) : (
                  <div className="cmp-pair">
                    <DistSide events={a.events} cycle={cycle} name={nameA} lastCycle={aligned.chosenA.length - 1} />
                    <DistSide events={b.events} cycle={cycle} name={nameB} lastCycle={aligned.chosenB.length - 1} />
                  </div>
                )}
              </>
            )
          })()}
          <h3>Cycle {cycle} · Attention</h3>
          {heads.length === 0 ? (
            <p className="fine attn-note">No detected heads recorded at this cycle.</p>
          ) : (
            <>
              <div className="head-chip-row">
                {heads.map((h) => (
                  <button key={`${h.layer}-${h.head}`} data-testid="cmp-head-chip"
                    data-active={String(selectedPair === h)} className={`head-chip pill${selectedPair === h ? ' is-picked' : ''}`}
                    onClick={() => setHeadKey(`${h.layer}-${h.head}`)}>
                    {h.a?.label ?? h.b?.label}{' '}
                    <span className="head-loc">L{h.layer}·H{h.head}</span>
                  </button>
                ))}
              </div>
              {selectedPair && (
                <div className="cmp-pair">
                  <AttnSide events={a.events} head={selectedPair.a} pair={selectedPair} name={nameA}
                    cycle={cycle} lastCycle={aligned.chosenA.length - 1} />
                  <AttnSide events={b.events} head={selectedPair.b} pair={selectedPair} name={nameB}
                    cycle={cycle} lastCycle={aligned.chosenB.length - 1} />
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  )
}

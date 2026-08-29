import { useState } from 'react'
import type { RunRecord } from '../../app/runsStore'
import type { AttentionHead, TokenInfo, TraceEvent } from '../../trace/types'
import { AttentionHeatmap } from '../AttentionHeatmap'
import { Thumb } from '../AttentionGridExplorer'
import { distributionFor, latestOfType, visibleTokens } from '../selectors'
import { alignRuns, pairedHeads, type PairedHead } from './compareSelectors'

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
          <span className="cmp-bar-token">{t.text.trim() || '·'}</span>
          <span className="cmp-bar" style={{ width: `${Math.max(2, Math.round(t.prob * 100))}%` }} />
          <span className="cmp-bar-prob">{Math.round(t.prob * 100)}%</span>
        </div>
      ))}
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
        <AttentionHeatmap heads={[head]} tokens={runTokens(events)} />
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
          <p className="attn-note">not among this run's detected heads at this cycle — run-level thumbnail shown
            (full matrices are kept only for each cycle's showcase heads)</p>
        </div>
      ) : (
        <p data-testid="cmp-fallback" className="cmp-fallback attn-note">not among this run's detected heads
          at this cycle — no full matrix captured</p>
      )}
    </div>
  )
}

export function CompareView({ a, b }: { a: RunRecord; b: RunRecord }) {
  const aligned = alignRuns(a.events, b.events)
  const [cycle, setCycle] = useState<number | null>(null)
  const [headKey, setHeadKey] = useState<string | null>(null)
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
        {prompt.map((t, i) => <span key={`p${i}`} className="cmp-token cmp-token-prompt">{t.text}</span>)}
        <span className="cmp-stream-divider" aria-hidden="true" />
        {chosen.map((t, c) => (
          <button key={`c${c}`} data-testid="cmp-token" data-fork={String(aligned.forkCycle === c)}
            data-selected={String(cycle === c)} className="cmp-token cmp-token-generated"
            title={`inspect cycle ${c}`} onClick={() => selectCycle(c)}>{t.text}</button>
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
        <MetaRow label="prompt" a={a.meta.prompt} b={b.meta.prompt} />
        <MetaRow label="T" a={String(a.meta.params.temperature)} b={String(b.meta.params.temperature)} />
        <MetaRow label="top-k" a={String(a.meta.params.topK)} b={String(b.meta.params.topK)} />
        <MetaRow label="max" a={String(a.meta.params.maxNewTokens)} b={String(b.meta.params.maxNewTokens)} />
        <MetaRow label="mode" a={a.meta.mode} b={b.meta.mode} />
        <MetaRow label="model" a={a.meta.modelId ?? '—'} b={b.meta.modelId ?? '—'} />
        <MetaRow label="ended" a={a.meta.reason} b={b.meta.reason} />
      </div>
      {!aligned.samePrompt && (
        <p data-testid="cmp-badge" className="cmp-badge">different prompts — aligned by generation cycle</p>
      )}
      {aligned.samePrompt && aligned.forkCycle === null
        && aligned.chosenA.length === aligned.chosenB.length && (
        <p data-testid="cmp-note" className="cmp-badge">identical outputs</p>
      )}
      {stream('a')}
      {stream('b')}
      <div className="cmp-ruler">
        {Array.from({ length: aligned.maxCycles }, (_, c) => (
          <button key={c} data-testid="cmp-tick" data-selected={String(cycle === c)}
            data-fork={String(aligned.forkCycle === c)} className="cmp-tick"
            onClick={() => selectCycle(c)}>{c}</button>
        ))}
      </div>
      {cycle !== null && (
        <>
          <h3>cycle {cycle} · distributions</h3>
          <div className="cmp-pair">
            <DistSide events={a.events} cycle={cycle} name={nameA} lastCycle={aligned.chosenA.length - 1} />
            <DistSide events={b.events} cycle={cycle} name={nameB} lastCycle={aligned.chosenB.length - 1} />
          </div>
          <h3>cycle {cycle} · attention</h3>
          {heads.length === 0 ? (
            <p className="attn-note">no detected heads recorded at this cycle</p>
          ) : (
            <>
              <div className="head-chip-row">
                {heads.map((h) => (
                  <button key={`${h.layer}-${h.head}`} data-testid="cmp-head-chip"
                    data-active={String(selectedPair === h)} className="head-chip"
                    onClick={() => setHeadKey(`${h.layer}-${h.head}`)}>
                    {h.a?.label ?? h.b?.label}
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

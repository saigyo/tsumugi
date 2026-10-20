import { useLayoutEffect, useRef, useState } from 'react'
import type { TraceEvent } from '../trace/types'
import { visibleToken } from './tokenText'
import { activeStage, eventAt, flowShapes, latestOfType, stageEventIndex, thousands, viewedCycle, type FlowShapes, type StageId } from './selectors'

// what travels along the connector INTO stage i (see FlowShapes)
const CONNECTOR_SHAPE: Array<keyof FlowShapes> = ['ids', 'ids', 'stream', 'lastRow', 'vocab', 'probs']

const STAGES: Array<{ id: Exclude<StageId, null>; label: string }> = [
  { id: 'tokenizer', label: 'Tokenizer' },
  { id: 'embeddings', label: 'Embeddings' },
  { id: 'layers', label: 'Layers' },
  { id: 'logits', label: 'Logits' },
  { id: 'softmax', label: 'Softmax' },
  { id: 'sampler', label: 'Sampler' },
]

function summaryFor(stage: Exclude<StageId, null>, events: TraceEvent[], cursor: number): string | null {
  switch (stage) {
    case 'tokenizer': {
      const e = latestOfType(events, cursor, 'tokenize')
      return e ? `${e.tokens.length} tokens` : null
    }
    case 'embeddings': {
      const e = latestOfType(events, cursor, 'embed')
      if (!e) return null
      const vocab = latestOfType(events, cursor, 'run-start')?.vocabSize
      return vocab ? `${thousands(vocab)} × ${e.dims}` : `${e.dims} dims`
    }
    case 'layers': {
      const e = latestOfType(events, cursor, 'layer')
      return e ? `${e.total} layers` : null
    }
    case 'logits': {
      const e = latestOfType(events, cursor, 'logits')
      return e ? `top: ${e.topK[0] ? visibleToken(e.topK[0]) : '—'}` : null
    }
    case 'softmax': {
      const e = latestOfType(events, cursor, 'softmax')
      const top = e?.topK[0]
      return top ? `${visibleToken(top)}: ${Math.round(top.prob * 100)}%` : null
    }
    case 'sampler': {
      const e = latestOfType(events, cursor, 'sample')
      return e ? `chose ${visibleToken(e.chosen)}` : null
    }
  }
}

// The loop's corners, in band pixels: out of the Sampler's right edge, down
// below the cards, left under the Embeddings card and up into it.
interface LoopGeometry { width: number; height: number; d: string; labelX: number; labelY: number }

function loopGeometry(band: HTMLElement): LoopGeometry | null {
  const card = (stage: string) => band.querySelector<HTMLElement>(`[data-stage="${stage}"]`)
  const from = card('sampler'), to = card('embeddings')
  if (!from || !to) return null
  const outX = from.offsetLeft + from.offsetWidth
  const outY = from.offsetTop + from.offsetHeight / 2
  const turnX = outX + 12
  const baseY = band.offsetHeight - 7
  const inX = to.offsetLeft + to.offsetWidth / 2
  const inY = to.offsetTop + to.offsetHeight + 2
  return {
    width: band.scrollWidth, height: band.offsetHeight,
    d: `M${outX + 2} ${outY} H${turnX} V${baseY} H${inX} V${inY}`,
    labelX: (inX + turnX) / 2, labelY: baseY,
  }
}

export function PipelineBand({ events, cursor, onStageClick }: {
  events: TraceEvent[]
  cursor: number
  onStageClick?: (index: number) => void
}) {
  const active = activeStage(eventAt(events, cursor))
  const shapes = flowShapes(events, cursor)
  // The tokenizer runs once, before the loop: later cycles start at the
  // Embeddings with the Sampler's id, so seeking back would rewind the run.
  const outsideLoop = viewedCycle(events, cursor) > 0
  const [noteOpen, setNoteOpen] = useState(false)
  if (noteOpen && !outsideLoop) setNoteOpen(false)
  const showNote = noteOpen && outsideLoop
  const bandRef = useRef<HTMLDivElement>(null)
  const [loop, setLoop] = useState<LoopGeometry | null>(null)
  useLayoutEffect(() => {
    const band = bandRef.current
    if (!band) return
    const measure = () => setLoop(loopGeometry(band))
    measure()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(measure)
    ro.observe(band)
    band.querySelectorAll('.stage-card').forEach((c) => ro.observe(c))
    return () => ro.disconnect()
  }, [])
  return (
    <>
    <div ref={bandRef} className="pipeline-band">
      {STAGES.map((s, i) => {
        const summary = summaryFor(s.id, events, cursor)
        const dimmed = s.id === 'tokenizer' && outsideLoop
        const target = stageEventIndex(events, cursor, s.id)
        const clickable = (dimmed || target >= 0) && onStageClick != null
        const activate = () => dimmed ? setNoteOpen((o) => !o) : onStageClick?.(target)
        const shape = i > 0 ? shapes[CONNECTOR_SHAPE[i]] : undefined
        return (
          <div key={s.id} className="stage-wrap" data-via-loop={s.id === 'embeddings' && outsideLoop ? 'true' : undefined}>
            {i > 0 && (
              <span className="stage-arrow-wrap">
                <span className="stage-arrow">→</span>
                {shape && <span data-testid="flow-shape" className="flow-shape">{shape}</span>}
              </span>
            )}
            <div data-testid="stage-card" data-stage={s.id} data-active={String(active === s.id)}
              data-clickable={String(clickable)} className="stage-card"
              {...(s.id === 'tokenizer' ? { 'data-outside-loop': String(outsideLoop) } : {})}
              role={clickable ? 'button' : undefined} tabIndex={clickable ? 0 : undefined}
              aria-expanded={dimmed ? showNote : undefined}
              title={dimmed ? 'Runs once, before the loop' : clickable ? `Jump to the ${s.label} step of the current token` : undefined}
              onClick={clickable ? activate : undefined}
              onKeyDown={clickable ? (e) => { if (e.key === 'Enter' || e.key === ' ') activate() } : undefined}>
              {s.label}
              {summary && <div data-testid="stage-summary" className="stage-summary">{summary}</div>}
            </div>
          </div>
        )
      })}
      <svg data-testid="loop-arrow" data-from="sampler" data-to="embeddings" className="loop-arrow"
        data-active={String(outsideLoop && active === 'embeddings')}
        width={loop?.width ?? 0} height={loop?.height ?? 0} aria-label="loop from the Sampler back to the Embeddings">
        <defs>
          <marker id="loop-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
          </marker>
        </defs>
        {loop && <path d={loop.d} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" markerEnd="url(#loop-head)" />}
      </svg>
      {shapes.loop && (
        <span data-testid="loop-label" className="loop-label"
          style={loop ? { left: loop.labelX, top: loop.labelY } : undefined}>{shapes.loop}</span>
      )}
    </div>
    {showNote && (
      <p data-testid="tokenizer-note" role="status" className="tokenizer-note">The tokenizer runs once, before the
        loop. Generated tokens come straight from the Sampler as IDs and are never tokenized.</p>
    )}
    </>
  )
}

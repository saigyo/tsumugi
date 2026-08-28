import { useEffect, useRef, useState } from 'react'
import type { TraceEvent } from '../trace/types'

type GridEvent = Extract<TraceEvent, { type: 'attention-grid' }>
type GridCell = GridEvent['cells'][number]
type SortKey = 'layer' | 'distinctive' | 'previous-token' | 'sink' | 'induction'

const SORTS: SortKey[] = ['layer', 'distinctive', 'previous-token', 'sink', 'induction']

const SCORE_OF: Record<Exclude<SortKey, 'layer'>, (c: GridCell) => number> = {
  distinctive: (c) => c.distinctiveScore,
  'previous-token': (c) => c.prevTokenScore,
  sink: (c) => c.sinkScore,
  induction: (c) => c.inductionScore ?? 0,
}

function topStat(c: GridCell): string {
  const entries: Array<[string, number]> = [
    ['prev-token', c.prevTokenScore],
    ['sink', c.sinkScore],
    ['induction', c.inductionScore ?? 0],
    ['distinctive', c.distinctiveScore],
  ]
  entries.sort((a, b) => b[1] - a[1])
  return `${entries[0][0]} ${entries[0][1].toFixed(2)}`
}

// One canvas pixel per thumb bucket, upscaled by CSS (image-rendering:
// pixelated). Canvas here is a recorded amendment to the v1 "SVG + CSS only"
// constraint: 270 SVG thumbnails would jank; the main heatmap stays SVG.
export function Thumb({ thumb }: { thumb: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return  // jsdom has no 2d context; structure still renders
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    thumb.forEach((row, r) => row.forEach((w, c) => {
      const v = Math.min(1, Math.max(0, w))
      ctx.fillStyle = `hsl(211 ${Math.round(30 + 25 * v)}% ${Math.round(94 - 70 * v)}%)`
      ctx.fillRect(c, r, 1, 1)
    }))
  }, [thumb])
  const dim = Math.max(thumb.length, 1)
  return <canvas ref={ref} width={dim} height={dim} className="grid-thumb" />
}

export function AttentionGridExplorer({ grid, onPin }: {
  grid: GridEvent
  onPin: (layer: number, head: number) => void
}) {
  const [sort, setSort] = useState<SortKey>('layer')

  const aggregate = (layer: number): number[][] => {
    const cells = grid.cells.filter((c) => c.layer === layer)
    const dim = cells[0]?.thumb.length ?? 0
    return Array.from({ length: dim }, (_, r) => Array.from({ length: dim }, (_, c) =>
      cells.reduce((sum, cell) => sum + (cell.thumb[r]?.[c] ?? 0), 0) / cells.length))
  }

  const cellButton = (c: GridCell) => (
    <button key={`${c.layer}-${c.head}`} data-testid="grid-cell" className="grid-cell"
      title={`L${c.layer}·H${c.head} — ${topStat(c)}`} onClick={() => onPin(c.layer, c.head)}>
      <Thumb thumb={c.thumb} />
      <span className="grid-loc">L{c.layer}·H{c.head}</span>
    </button>
  )

  return (
    <div data-testid="grid-explorer" className="grid-explorer">
      <div className="grid-explorer-header">
        <span className="grid-semantics">attention accumulated over the whole run</span>
        <label className="grid-sort-label">sort{' '}
          <select data-testid="grid-sort" value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}>
            {SORTS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
      </div>
      {sort === 'layer' ? (
        <div className="grid-scroll">
          {Array.from({ length: grid.layers }, (_, l) => (
            <div key={l} data-testid="grid-row" className="grid-row">
              <div data-testid="grid-aggregate" className="grid-aggregate" title={`L${l} — mean of heads`}>
                <Thumb thumb={aggregate(l)} />
                <span className="grid-loc">L{l} ∅</span>
              </div>
              {grid.cells.filter((c) => c.layer === l).map(cellButton)}
            </div>
          ))}
        </div>
      ) : (
        <div className="grid-scroll grid-flat">
          {[...grid.cells].sort((a, b) => SCORE_OF[sort](b) - SCORE_OF[sort](a)).map(cellButton)}
        </div>
      )}
    </div>
  )
}

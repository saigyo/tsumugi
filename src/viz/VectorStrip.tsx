import { cellFill, peakOf } from './stripMath'

// One vector as a row of coloured cells (see stripMath for the palette).
export function VectorStrip({ cells, cellW, cellH, gap = 1, testId, cellTestId, ariaLabel, titleFor, className, fit }: {
  cells: number[]
  cellW: number
  cellH: number
  gap?: number
  testId?: string
  cellTestId?: string
  ariaLabel: string
  titleFor?: (index: number, value: number) => string
  className?: string
  // shrink to the width of the column (never above the natural width), the cells scaling with it
  fit?: boolean
}) {
  const peak = peakOf(cells)
  const w = cells.length * cellW
  return (
    <svg data-testid={testId} className={className} width={fit ? '100%' : w} height={cellH} viewBox={`0 0 ${w} ${cellH}`} role="img" aria-label={ariaLabel}
      {...(fit ? { preserveAspectRatio: 'none', style: { maxWidth: w, display: 'block' } } : {})}>
      {cells.map((v, c) => (
        <rect key={c} data-testid={cellTestId} x={c * cellW} y={0} width={cellW - gap} height={cellH} style={{ fill: cellFill(v, peak) }}>
          {titleFor && <title>{titleFor(c, v)}</title>}
        </rect>
      ))}
    </svg>
  )
}

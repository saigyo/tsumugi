import { cellFill, peakOf } from './stripMath'

// One vector as a row of coloured cells (see stripMath for the palette).
export function VectorStrip({ cells, cellW, cellH, gap = 1, testId, cellTestId, ariaLabel, titleFor, className }: {
  cells: number[]
  cellW: number
  cellH: number
  gap?: number
  testId?: string
  cellTestId?: string
  ariaLabel: string
  titleFor?: (index: number, value: number) => string
  className?: string
}) {
  const peak = peakOf(cells)
  const w = cells.length * cellW
  return (
    <svg data-testid={testId} className={className} width={w} height={cellH} viewBox={`0 0 ${w} ${cellH}`} role="img" aria-label={ariaLabel}>
      {cells.map((v, c) => (
        <rect key={c} data-testid={cellTestId} x={c * cellW} y={0} width={cellW - gap} height={cellH} fill={cellFill(v, peak)}>
          {titleFor && <title>{titleFor(c, v)}</title>}
        </rect>
      ))}
    </svg>
  )
}

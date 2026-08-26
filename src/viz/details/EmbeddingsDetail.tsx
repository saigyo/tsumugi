import type { TraceEvent } from '../../trace/types'

export function EmbeddingsDetail({ event }: { event: Extract<TraceEvent, { type: 'embed' }> }) {
  return (
    <div data-testid="detail-embeddings" className="detail">
      <h3>Embeddings</h3>
      <p>Each token becomes a vector of {event.dims} numbers (showing 16 dims of the last {event.preview.length} tokens).</p>
      <svg width={event.preview[0]?.length * 14} height={event.preview.length * 14}>
        {event.preview.map((row, r) =>
          row.map((v, c) => (
            <rect key={`${r}-${c}`} x={c * 14} y={r * 14} width={12} height={12}
              fill={`hsl(${v >= 0 ? 210 : 10} 70% ${50 + Math.abs(v) * 30}%)`} />
          )),
        )}
      </svg>
    </div>
  )
}

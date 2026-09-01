import type { TraceEvent } from '../../trace/types'

export function EmbeddingsDetail({ event }: { event: Extract<TraceEvent, { type: 'embed' }> }) {
  return (
    <div data-testid="detail-embeddings" className="detail">
      <h3>Embeddings</h3>
      <p>Each token becomes a vector of {event.dims} numbers.</p>
    </div>
  )
}

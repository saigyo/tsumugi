import { useState } from 'react'
import { coversModel } from '../../geometry/asset'
import { useGeometry } from '../../geometry/useGeometry'
import type { TraceEvent } from '../../trace/types'
import { EmbeddingLookup } from '../EmbeddingLookup'
import { embeddingRows, latestOfType, thousands } from '../selectors'

// Container: resolves where vectors come from (exact trace rows, else the
// geometry asset by id) and owns the selected position. Pure function of
// (events, cursor, asset) apart from that one piece of UI state.
export function EmbeddingsDetail({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const embed = latestOfType(events, cursor, 'embed')
  const runStart = latestOfType(events, cursor, 'run-start')
  const { tokens, rows, source } = embeddingRows(events, cursor)
  // snaps to the newest token when unset, or when scrubbing back past the pick
  const [picked, setPicked] = useState<number | null>(null)
  const selected = picked == null || picked >= tokens.length ? tokens.length - 1 : picked
  const geo = useGeometry()
  const covered = geo.status === 'ready' && geo.asset != null && runStart != null
    && coversModel(geo.asset.manifest, runStart.modelId)
  const asset = covered ? geo.asset : undefined
  const pending = geo.status === 'loading'
  if (!embed) return null
  const dims = embed.dims
  const vocab = runStart?.vocabSize
  const vectorFor = (pos: number): ArrayLike<number> | undefined =>
    rows ? rows[pos] : asset?.vector(tokens[pos].id)
  return (
    <div data-testid="detail-embeddings" className="detail">
      <h3>Embeddings</h3>
      <p>
        A lookup, not a computation. Each token id selects one row of a learned matrix{' '}
        <code>E [{vocab ? thousands(vocab) : '?'} × {dims}]</code>; the rows stacked up are{' '}
        <code>x [{tokens.length} × {dims}]</code>.
      </p>
      <EmbeddingLookup tokens={tokens} dims={dims} vocabSize={vocab} selected={selected} onSelect={setPicked}
        vectorFor={vectorFor} source={source}
        missingNote={pending ? 'loading vocabulary geometry…' : 'vector values unavailable offline'} />
    </div>
  )
}

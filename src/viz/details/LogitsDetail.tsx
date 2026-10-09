import { useState } from 'react'
import { coversModel } from '../../geometry/asset'
import { useGeometry } from '../../geometry/useGeometry'
import type { TraceEvent } from '../../trace/types'
import { LogitsReadout } from '../LogitsReadout'
import { latestOfType, thousands } from '../selectors'

// Container: picks the cycle's logits/softmax events, resolves asset rows for
// sim and legacy traces, and owns the selected candidate (snaps to the top
// candidate when the cursor moves to another cycle).
export function LogitsDetail({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const logits = latestOfType(events, cursor, 'logits')
  const softmax = latestOfType(events, cursor, 'softmax')
  const embed = latestOfType(events, cursor, 'embed')
  const runStart = latestOfType(events, cursor, 'run-start')
  const [pick, setPick] = useState<{ cycle: number; index: number } | null>(null)
  const geo = useGeometry()
  if (!logits) return null
  const showProbs = softmax != null && softmax.cycle === logits.cycle
  const dims = embed?.dims ?? logits.lastHidden?.length ?? 576
  const vocabSize = runStart?.vocabSize
  const selected = pick && pick.cycle === logits.cycle && pick.index < logits.topK.length ? pick.index : 0
  const covered = geo.status === 'ready' && geo.asset != null && runStart != null
    && coversModel(geo.asset.manifest, runStart.modelId)
  const asset = covered ? geo.asset : undefined
  const assetRowFor = (id: number) =>
    asset && id >= 0 && id < asset.manifest.vocabSize ? asset.vector(id) : undefined
  const assetNote = geo.status === 'loading' ? 'loading vocabulary geometry…' : 'vector values unavailable'
  return (
    <div data-testid="detail-logits" className="detail">
      <h3>{showProbs ? `Softmax (T=${softmax!.temperature})` : 'Logits'}</h3>
      <p>{showProbs
        ? 'Softmax turns scores into probabilities that sum to 100%.'
        : 'Each score is a dot product: the last token’s final vector against one row of E — the same matrix the Embeddings stage looked rows up in.'}</p>
      {vocabSize && (
        <p data-testid="logits-formula" className="logits-formula">
          logits = x_last · Eᵀ&emsp;[1×{dims}] · [{dims}×{thousands(vocabSize)}] → [{thousands(vocabSize)}]
        </p>
      )}
      <LogitsReadout candidates={logits.topK} probs={showProbs ? softmax!.topK.map((c) => c.prob) : undefined}
        lastHidden={logits.lastHidden} dims={dims} vocabSize={vocabSize}
        selected={selected} onSelect={(index) => setPick({ cycle: logits.cycle, index })}
        assetRowFor={assetRowFor} assetNote={assetNote} />
    </div>
  )
}

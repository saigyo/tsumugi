import { useState } from 'react'
import { coversModel } from '../../geometry/asset'
import { useGeometry } from '../../geometry/useGeometry'
import type { TraceEvent } from '../../trace/types'
import { LogitsReadout } from '../LogitsReadout'
import { latestOfType } from '../selectors'
import { Term } from '../../glossary/Term'
import { Formula } from '../Formula'
import { logitsTex } from '../tex'

// Container: picks the cycle's logits event, resolves asset rows for sim and
// legacy traces, and owns the selected candidate (snaps to the top candidate
// when the cursor moves to another cycle). The softmax step has its own card.
export function LogitsDetail({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const logits = latestOfType(events, cursor, 'logits')
  const embed = latestOfType(events, cursor, 'embed')
  const runStart = latestOfType(events, cursor, 'run-start')
  const [pick, setPick] = useState<{ cycle: number; index: number } | null>(null)
  const geo = useGeometry()
  if (!logits) return null
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
      <h3>Logits <Term id="logit" variant="icon" label="About this stage" testId="stage-info">ⓘ</Term></h3>
      <p>Each <Term id="logit">score</Term> is a dot product: the last token’s final vector against one row of E — the same matrix the Embeddings stage looked rows up in.</p>
      {vocabSize && (
        <Formula testId="logits-formula" className="card-formula" inline tex={logitsTex(dims, vocabSize)} />
      )}
      <LogitsReadout candidates={logits.topK} lastHidden={logits.lastHidden} dims={dims} vocabSize={vocabSize}
        selected={selected} onSelect={(index) => setPick({ cycle: logits.cycle, index })}
        assetRowFor={assetRowFor} assetNote={assetNote} />
    </div>
  )
}

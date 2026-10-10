import { coversModel } from '../../geometry/asset'
import { useGeometry } from '../../geometry/useGeometry'
import type { TraceEvent } from '../../trace/types'
import { LogitsReadout } from '../LogitsReadout'
import { latestOfType } from '../selectors'
import { Term } from '../../glossary/Term'
import { Formula } from '../Formula'
import { logitsTex } from '../tex'
import { selectedCandidate, useCandidateSelection } from '../candidateSelection'

// Container: picks the cycle's logits event, resolves asset rows for sim and
// legacy traces, and shows the selected candidate (shared with the votes card
// through candidateSelection; snaps to the top candidate on another cycle).
// The softmax step has its own card.
export function LogitsDetail({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const logits = latestOfType(events, cursor, 'logits')
  const embed = latestOfType(events, cursor, 'embed')
  const runStart = latestOfType(events, cursor, 'run-start')
  const pick = useCandidateSelection((s) => s.pick)
  const select = useCandidateSelection((s) => s.select)
  const geo = useGeometry()
  if (!logits) return null
  const dims = embed?.dims ?? logits.lastHidden?.length ?? 576
  const vocabSize = runStart?.vocabSize
  const selected = selectedCandidate(pick, logits.cycle, logits.topK.length)
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
        selected={selected} onSelect={(index) => select(logits.cycle, index)}
        assetRowFor={assetRowFor} assetNote={assetNote} />
    </div>
  )
}

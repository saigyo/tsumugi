import { STAGES } from './PipelineBand'
import type { StageId } from './selectors'

// "Stage n of 6": where the card below sits in the pipeline band's own list.
export function StagePill({ stage }: { stage: Exclude<StageId, null> }) {
  const n = STAGES.findIndex((s) => s.id === stage) + 1
  return <span data-testid="stage-pill" className="stage-pill">Stage {n} of {STAGES.length}</span>
}

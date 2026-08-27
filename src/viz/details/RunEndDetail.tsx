import type { TraceEvent } from '../../trace/types'

type RunStartEvent = Extract<TraceEvent, { type: 'run-start' }>
type RunEndEvent = Extract<TraceEvent, { type: 'run-end' }>

const REASON_LABEL: Record<RunEndEvent['reason'], string> = {
  eos: 'the model chose its end-of-sequence token',
  'max-tokens': 'the max-tokens limit was reached',
  aborted: 'the run was aborted',
  error: 'the run failed',
}

export function RunEndDetail({ runStart, runEnd, promptTokens, generatedTokens }: {
  runStart?: RunStartEvent
  runEnd: RunEndEvent
  promptTokens: number
  generatedTokens: number
}) {
  return (
    <div data-testid="detail-run-end" className="detail">
      <h3>Run finished ({runEnd.reason})</h3>
      <p>
        Generation stopped because {REASON_LABEL[runEnd.reason]}
        {runEnd.message ? `: ${runEnd.message}` : '.'}
      </p>
      <p>
        {promptTokens} prompt tokens in, {generatedTokens} tokens generated
        {runStart && ` (T=${runStart.params.temperature}, top-k ${runStart.params.topK}, max ${runStart.params.maxNewTokens})`}.
      </p>
      <p>Scrub back to inspect any step, or press Generate for a new run.</p>
    </div>
  )
}

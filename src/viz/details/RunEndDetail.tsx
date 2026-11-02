import type { TraceEvent } from '../../trace/types'
import { continueLimit, type ShareSettings } from '../../app/shareLink'
import { useRunRequest } from '../runRequest'
import { ShareLinkButton } from '../ShareLinkButton'

type RunStartEvent = Extract<TraceEvent, { type: 'run-start' }>
type RunEndEvent = Extract<TraceEvent, { type: 'run-end' }>

const REASON_LABEL: Record<RunEndEvent['reason'], string> = {
  eos: 'the model chose its end-of-sequence token',
  'max-tokens': 'the max-tokens limit was reached',
  aborted: 'the run was aborted',
  error: 'the run failed',
}

export function RunEndDetail({ runStart, runEnd, promptTokens, generatedTokens, shareable, truncated }: {
  runStart?: RunStartEvent
  runEnd: RunEndEvent
  promptTokens: number
  generatedTokens: number
  shareable?: ShareSettings | null   // shareableRun(events); null for runs that cannot be repeated
  truncated?: boolean
}) {
  const { model, busy, continueRun } = useRunRequest()
  const limit = shareable ? continueLimit(shareable.params.maxNewTokens) : null
  const canContinue = shareable && limit !== null && runEnd.reason === 'max-tokens' && !truncated
  const wanted = shareable?.rev ?? null
  const showSeed = runStart?.mode === 'real' && runStart.seed !== undefined
    && (runEnd.reason === 'eos' || runEnd.reason === 'max-tokens')
  const wrongModel = !model || model.revision !== wanted
  return (
    <div data-testid="detail-run-end" className="detail">
      <h3>Run finished ({runEnd.reason})</h3>
      <p>
        Generation stopped because {REASON_LABEL[runEnd.reason]}
        {runEnd.message ? `: ${runEnd.message}` : '.'}
      </p>
      <p>
        {promptTokens} prompt tokens in, {generatedTokens} tokens generated
        {runStart && ` (T=${runStart.params.temperature}, top-k ${runStart.params.topK}, max ${runStart.params.maxNewTokens}${showSeed ? `, seed ${runStart.seed}` : ''})`}.
      </p>
      {shareable && (
        <p className="run-end-actions">
          {canContinue && (
            <button type="button" className="knockout-run" disabled={busy || wrongModel || !continueRun}
              title={wrongModel ? `Load model version ${wanted ?? 'stock'} to continue this run` : undefined}
              onClick={() => continueRun?.(limit)}>Run again with {limit} tokens</button>
          )}
          <ShareLinkButton settings={shareable} />
        </p>
      )}
      <p>Scrub back to inspect any step, or press Generate for a new run.</p>
    </div>
  )
}

import { useState, type ReactNode } from 'react'
import type { GenParams, Mode } from '../trace/types'

export interface PromptExample {
  id: string
  label: string
  prompt: string
  hint: string
}

export interface PromptBarProps {
  mode: Mode
  onModeChange(mode: Mode): void
  onGenerate(prompt: string, params: GenParams): void
  busy: boolean
  // a run is being computed: Generate shows activity and a Stop button appears
  generating?: boolean
  onStop?(): void
  examples?: PromptExample[]
  status?: ReactNode
}

const clampOrKeep = (raw: string, min: number, max: number, prev: number): number => {
  const n = Number(raw)
  if (raw.trim() === '' || Number.isNaN(n)) return prev
  return Math.min(max, Math.max(min, n))
}

export function PromptBar({ mode, onModeChange, onGenerate, busy, generating = false, onStop, examples, status }: PromptBarProps) {
  const [prompt, setPrompt] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(10)
  const [maxNewTokens, setMaxNewTokens] = useState(20)

  return (
    <div className="prompt-bar">
      <div className="prompt-row">
        <input data-testid="prompt-input" value={prompt} placeholder="Type a prompt…"
          onChange={(e) => setPrompt(e.target.value)} />
        <button data-testid="btn-generate" disabled={prompt.trim() === '' || busy || generating}
          aria-busy={generating || undefined}
          onClick={() => onGenerate(prompt, { temperature, topK, maxNewTokens })}>
          {generating
            ? <><span data-testid="generate-activity" className="generate-activity" aria-hidden="true" />Generating…</>
            : 'Generate'}
        </button>
        {generating && onStop && (
          <button data-testid="btn-stop" className="btn-stop" onClick={onStop}
            title="Abort the run in flight; what was generated so far is kept">
            Stop
          </button>
        )}
      </div>
      <div className="config-row">
        <label title="Temperature: divides the logits before softmax. Below 1 sharpens the distribution toward the favorite (safer, more repetitive); above 1 flattens it (more varied, more error-prone); exactly 0 always picks the top candidate (greedy). It changes how concentrated the odds are, never the ranking.">
          T <input data-testid="temp-input" type="number" step="0.1" min="0" max="2" value={temperature}
          onChange={(e) => setTemperature(clampOrKeep(e.target.value, 0, 2, temperature))} /></label>
        <label title="Top-k: only the k highest-scoring candidates are allowed into the draw; everything else gets probability zero. Cuts off the long tail of weird tokens that would otherwise fire eventually. k=1 is greedy decoding regardless of temperature.">
          top-k <input data-testid="topk-input" type="number" min="1" max="10" value={topK}
          onChange={(e) => setTopK(clampOrKeep(e.target.value, 1, 10, topK))} /></label>
        <label title="Max new tokens: the loop bound — the generation cycle runs at most this many times. The run ends earlier if the model samples its end-of-sequence token (the run summary shows which happened). In real mode every token is one full forward pass.">
          max <input data-testid="maxtok-input" type="number" min="1" max="100" value={maxNewTokens}
          onChange={(e) => setMaxNewTokens(clampOrKeep(e.target.value, 1, 100, maxNewTokens))} /></label>
        <span className="config-divider" aria-hidden="true" />
        <label title="Runs HuggingFaceTB/SmolLM2-135M-Instruct in your browser. Downloads ~120 MB once on first use; cached afterward.">
          <input data-testid="mode-toggle" type="checkbox" checked={mode === 'real'}
            onChange={(e) => onModeChange(e.target.checked ? 'real' : 'sim')} />
          Real model <span className="mode-note">~120 MB</span>
        </label>
        {status && <span data-testid="model-status-slot" className="model-status-slot">{status}</span>}
      </div>
      {examples && examples.length > 0 && (
        <div className="example-chip-row">
          <span className="example-chip-label">Try:</span>
          {examples.map((ex) => (
            <button key={ex.id} data-testid="example-chip" className="example-chip" title={ex.hint}
              disabled={busy || generating}
              onClick={() => {
                setPrompt(ex.prompt)
                onGenerate(ex.prompt, { temperature, topK, maxNewTokens })
              }}>
              {ex.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

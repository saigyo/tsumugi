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
  examples?: PromptExample[]
  status?: ReactNode
}

const clampOrKeep = (raw: string, min: number, max: number, prev: number): number => {
  const n = Number(raw)
  if (raw.trim() === '' || Number.isNaN(n)) return prev
  return Math.min(max, Math.max(min, n))
}

export function PromptBar({ mode, onModeChange, onGenerate, busy, examples, status }: PromptBarProps) {
  const [prompt, setPrompt] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(10)
  const [maxNewTokens, setMaxNewTokens] = useState(20)

  return (
    <div className="prompt-bar">
      <div className="prompt-row">
        <input data-testid="prompt-input" value={prompt} placeholder="Type a prompt…"
          onChange={(e) => setPrompt(e.target.value)} />
        <button data-testid="btn-generate" disabled={prompt.trim() === '' || busy}
          onClick={() => onGenerate(prompt, { temperature, topK, maxNewTokens })}>
          Generate
        </button>
      </div>
      <div className="config-row">
        <label>T <input data-testid="temp-input" type="number" step="0.1" min="0" max="2" value={temperature}
          onChange={(e) => setTemperature(clampOrKeep(e.target.value, 0, 2, temperature))} /></label>
        <label>top-k <input data-testid="topk-input" type="number" min="1" max="10" value={topK}
          onChange={(e) => setTopK(clampOrKeep(e.target.value, 1, 10, topK))} /></label>
        <label>max <input data-testid="maxtok-input" type="number" min="1" max="100" value={maxNewTokens}
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
              disabled={busy}
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

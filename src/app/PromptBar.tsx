import { useState } from 'react'
import type { GenParams, Mode } from '../trace/types'

export interface PromptBarProps {
  mode: Mode
  onModeChange(mode: Mode): void
  onGenerate(prompt: string, params: GenParams): void
  busy: boolean
}

export function PromptBar({ mode, onModeChange, onGenerate, busy }: PromptBarProps) {
  const [prompt, setPrompt] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(10)
  const [maxNewTokens, setMaxNewTokens] = useState(20)

  return (
    <div className="prompt-bar">
      <input data-testid="prompt-input" value={prompt} placeholder="Type a prompt…"
        onChange={(e) => setPrompt(e.target.value)} />
      <label>
        <input data-testid="mode-toggle" type="checkbox" checked={mode === 'real'}
          onChange={(e) => onModeChange(e.target.checked ? 'real' : 'sim')} />
        Real model (~120 MB download on first use)
      </label>
      <label>T <input data-testid="temp-input" type="number" step="0.1" min="0" max="2" value={temperature}
        onChange={(e) => setTemperature(Number(e.target.value))} /></label>
      <label>top-k <input data-testid="topk-input" type="number" min="1" max="10" value={topK}
        onChange={(e) => setTopK(Number(e.target.value))} /></label>
      <label>max <input data-testid="maxtok-input" type="number" min="1" max="100" value={maxNewTokens}
        onChange={(e) => setMaxNewTokens(Number(e.target.value))} /></label>
      <button data-testid="btn-generate" disabled={prompt.trim() === '' || busy}
        onClick={() => onGenerate(prompt, { temperature, topK, maxNewTokens })}>
        Generate
      </button>
    </div>
  )
}

import { useState } from 'react'
import { DEFAULT_SYSTEM_PROMPT, TEXT_FORMAT, systemOf } from '../engine/chatTemplate'
import { Term } from '../glossary/Term'
import type { GenParams, Mode, PromptFormat } from '../trace/types'

const REAL_TITLE = 'Runs SmolLM2-135M-Instruct in your browser from a custom export with attention and readout outputs (its output layer stays full precision). Downloads ~250 MB once on first use; cached afterward. If that fails it falls back to the ~120 MB stock export without them.'

export interface PromptExample {
  id: string
  label: string
  prompt: string
  hint: string
  format?: 'chat' | 'text'
}

export interface PromptBarProps {
  mode: Mode
  onModeChange(mode: Mode): void
  onGenerate(prompt: string, params: GenParams, format: PromptFormat): void
  busy: boolean
  // a run is being computed: Generate shows activity and a Stop button appears
  generating?: boolean
  onStop?(): void
  // what the activity label says while generating (a knockout sweep says Measuring…)
  activity?: string
  examples?: PromptExample[]
}

const clampOrKeep = (raw: string, min: number, max: number, prev: number): number => {
  const n = Number(raw)
  if (raw.trim() === '' || Number.isNaN(n)) return prev
  return Math.min(max, Math.max(min, n))
}

export function PromptBar({ mode, onModeChange, onGenerate, busy, generating = false, onStop, activity = 'Generating…', examples }: PromptBarProps) {
  const [prompt, setPrompt] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(10)
  const [maxNewTokens, setMaxNewTokens] = useState(20)
  const [chatFmt, setChatFmt] = useState(true)
  const [system, setSystem] = useState(DEFAULT_SYSTEM_PROMPT)
  const chatOn = mode === 'real' && chatFmt
  const format: PromptFormat = chatOn ? { kind: 'chat', system: systemOf(system) } : TEXT_FORMAT

  return (
    <div className="prompt-bar">
      <div className="prompt-row">
        <div className="seg" role="group" aria-label="Model" data-testid="mode-switch">
          <button type="button" data-testid="mode-sim" aria-pressed={mode === 'sim'}
            onClick={() => onModeChange('sim')}>Simulated</button>
          <button type="button" data-testid="mode-real" aria-pressed={mode === 'real' && !busy}
            aria-busy={(mode === 'real' && busy) || undefined} title={REAL_TITLE}
            onClick={() => { if (mode !== 'real') onModeChange('real') }}>Real model <span className="mode-note">· 250 MB</span></button>
        </div>
        {mode === 'real' && (
          <div className="seg" role="group" aria-label="Prompt format" data-testid="format-switch">
            <button type="button" data-testid="format-chat" aria-pressed={chatFmt} onClick={() => setChatFmt(true)}>Chat</button>
            <button type="button" data-testid="format-text" aria-pressed={!chatFmt} onClick={() => setChatFmt(false)}>Continue text</button>
            <Term id="chat-template" variant="icon" label="About the chat template" testId="format-info">ⓘ</Term>
          </div>
        )}
        <input data-testid="prompt-input" value={prompt} placeholder={chatOn ? 'Ask a question…' : 'Type a prompt…'}
          onChange={(e) => setPrompt(e.target.value)} />
        <button data-testid="btn-generate" disabled={prompt.trim() === '' || busy || generating}
          aria-busy={generating || undefined}
          onClick={() => onGenerate(prompt, { temperature, topK, maxNewTokens }, format)}>
          {generating
            ? <><span data-testid="generate-activity" className="generate-activity" aria-hidden="true" />{activity}</>
            : 'Generate'}
        </button>
        {generating && onStop && (
          <button data-testid="btn-stop" className="btn-stop" onClick={onStop}
            title="Abort the run in flight; what was generated so far is kept">
            Stop
          </button>
        )}
      </div>
      {chatOn && (
        <div className="system-row">
          <label htmlFor="system-input">System prompt</label>
          <input id="system-input" data-testid="system-input" value={system} onChange={(e) => setSystem(e.target.value)} />
          {system !== DEFAULT_SYSTEM_PROMPT && (
            <button type="button" data-testid="system-reset" className="link-button"
              onClick={() => setSystem(DEFAULT_SYSTEM_PROMPT)}>Reset to default</button>
          )}
        </div>
      )}
      <div className="config-row">
        <span className="config-field">
          <label>Temperature <input data-testid="temp-input" type="number" step="0.1" min="0" max="2" value={temperature}
            onChange={(e) => setTemperature(clampOrKeep(e.target.value, 0, 2, temperature))} /></label>
          <Term id="generation-settings" section="temperature" variant="icon" label="About temperature" testId="setting-info"
            tip="0 = always the top token; higher = more varied.">ⓘ</Term>
        </span>
        <span className="config-field">
          <label>Top-k <input data-testid="topk-input" type="number" min="1" max="10" value={topK}
            onChange={(e) => setTopK(clampOrKeep(e.target.value, 1, 10, topK))} /></label>
          <Term id="generation-settings" section="top-k" variant="icon" label="About top-k" testId="setting-info"
            tip="Only the k best candidates can be drawn.">ⓘ</Term>
        </span>
        <span className="config-field">
          <label>Max new tokens <input data-testid="maxtok-input" type="number" min="1" max="100" value={maxNewTokens}
            onChange={(e) => setMaxNewTokens(clampOrKeep(e.target.value, 1, 100, maxNewTokens))} /></label>
          <Term id="generation-settings" section="max-new-tokens" variant="icon" label="About max new tokens" testId="setting-info"
            tip="The run stops here at the latest.">ⓘ</Term>
        </span>
      </div>
      {examples && examples.length > 0 && (
        <div className="example-chip-row">
          <span className="example-chip-label">Try:</span>
          {examples.map((ex) => (
            <button key={ex.id} data-testid="example-chip" className="example-chip" title={ex.hint}
              disabled={busy || generating}
              onClick={() => {
                setPrompt(ex.prompt)
                const params = { temperature, topK, maxNewTokens }
                if (ex.format === 'chat') {
                  setChatFmt(true); setSystem(DEFAULT_SYSTEM_PROMPT)
                  onGenerate(ex.prompt, params, { kind: 'chat', system: DEFAULT_SYSTEM_PROMPT })
                } else {
                  setChatFmt(false)
                  onGenerate(ex.prompt, params, TEXT_FORMAT)
                }
              }}>
              {ex.format === 'chat' && <span data-testid="example-chip-tag" className="example-chip-tag">Chat</span>}
              {ex.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

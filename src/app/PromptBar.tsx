import { useState } from 'react'
import { DEFAULT_SYSTEM_PROMPT, TEXT_FORMAT, systemOf } from '../engine/chatTemplate'
import { Term } from '../glossary/Term'
import { parseSeed, type SharedRun } from './shareLink'
import type { GenParams, Mode, PromptFormat } from '../trace/types'
import { Icon } from '../ui/Icon'
import { decimalValue, normaliseDecimal, stepDecimal } from '../ui/decimalField'

const REAL_TITLE = 'Runs SmolLM2-135M-Instruct in your browser from a custom export with attention and readout outputs (its output layer stays full precision). Downloads ≈ 250 MB once on first use; cached afterward. If that fails it falls back to the ≈ 120 MB stock export without them.'

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
  onGenerate(prompt: string, params: GenParams, format: PromptFormat, seed?: number): void
  busy: boolean
  // the real model has been loaded in this session, whichever mode is chosen now (the download size is no longer news)
  realLoaded?: boolean
  // a run is being computed: Generate shows activity and a Stop button appears
  generating?: boolean
  onStop?(): void
  // what the activity label says while generating (a knockout sweep says Measuring…)
  activity?: string
  examples?: PromptExample[]
  // values from a share link, read once when the bar mounts
  initial?: SharedRun
  // the shelf holds saved runs: the examples row then starts closed
  hasRuns: boolean
}

const clampOrKeep = (raw: string, min: number, max: number, prev: number): number => {
  const n = Number(raw)
  if (raw.trim() === '' || Number.isNaN(n)) return prev
  return Math.min(max, Math.max(min, n))
}

export function PromptBar({ mode, onModeChange, onGenerate, busy, realLoaded = false, generating = false, onStop, activity = 'Generating…', examples, initial, hasRuns }: PromptBarProps) {
  const [prompt, setPrompt] = useState(initial?.prompt ?? '')
  const [temperature, setTemperature] = useState(initial?.params?.temperature ?? 0.8)
  const [tempText, setTempText] = useState(String(temperature))
  const [topK, setTopK] = useState(initial?.params?.topK ?? 10)
  const [maxNewTokens, setMaxNewTokens] = useState(initial?.params?.maxNewTokens ?? 20)
  const [chatFmt, setChatFmt] = useState(initial?.format?.kind !== 'text')
  const [system, setSystem] = useState(initial?.format?.kind === 'chat' ? (initial.format.system ?? '') : DEFAULT_SYSTEM_PROMPT)
  const [seedText, setSeedText] = useState(initial?.seed === undefined ? '' : String(initial.seed))
  const [ranOnce, setRanOnce] = useState(false)
  const [examplesToggled, setExamplesToggled] = useState<boolean | null>(null)
  const [systemToggled, setSystemToggled] = useState<boolean | null>(null)
  const chatOn = mode === 'real' && chatFmt
  // the real model is chosen and loaded: its option is the pressed one
  const realReady = mode === 'real' && !busy
  const seed = mode === 'real' ? parseSeed(seedText) ?? undefined : undefined
  const examplesOpen = examplesToggled ?? (!hasRuns && !ranOnce)
  const systemEdited = system !== DEFAULT_SYSTEM_PROMPT
  // latched at mount: a share link with a custom system prompt opens the row, and later resets do not close it
  const [systemOpenAtMount] = useState(initial?.format?.kind === 'chat' && initial.format.system !== DEFAULT_SYSTEM_PROMPT)
  const systemOpen = systemToggled ?? systemOpenAtMount
  const hasExamples = !!examples && examples.length > 0
  const generate: PromptBarProps['onGenerate'] = (p, params, f, sd) => {
    setRanOnce(true)
    if (sd === undefined) onGenerate(p, params, f)
    else onGenerate(p, params, f, sd)
  }
  const format: PromptFormat = chatOn ? { kind: 'chat', system: systemOf(system) } : TEXT_FORMAT

  return (
    <div className="prompt-box">
      <div className="prompt-line">
        <input data-testid="prompt-input" aria-label={chatOn ? 'Your message' : 'Prompt'} value={prompt} placeholder={chatOn ? 'Ask a question…' : 'Type a prompt…'}
          onChange={(e) => setPrompt(e.target.value)} />
        {generating && onStop && (
          <button data-testid="btn-stop" className="btn btn-danger btn-lg" onClick={onStop}
            title="Abort the run in flight; what was generated so far is kept">
            Stop
          </button>
        )}
        <button data-testid="btn-generate" className="btn btn-accent btn-lg" disabled={prompt.trim() === '' || busy || generating}
          aria-busy={generating || undefined}
          onClick={() => generate(prompt, { temperature, topK, maxNewTokens }, format, seed)}>
          {generating
            ? <><span data-testid="generate-activity" className="generate-activity" aria-hidden="true" />{activity}</>
            : 'Generate'}
        </button>
      </div>
      <div className="settings-line">
        <div className="seg" role="group" aria-label="Model" data-testid="mode-switch">
          <button type="button" data-testid="mode-sim" className={`seg-opt${mode === 'sim' ? ' is-view' : ''}`} aria-pressed={mode === 'sim'}
            onClick={() => onModeChange('sim')}>Simulated</button>
          <button type="button" data-testid="mode-real" className={`seg-opt${realReady ? ' is-view' : ''}`} aria-pressed={realReady}
            aria-busy={(mode === 'real' && busy) || undefined} title={REAL_TITLE}
            onClick={() => { if (mode !== 'real') onModeChange('real') }}>Real model{!realReady && !realLoaded && <span className="mode-note"> · 250 MB</span>}</button>
        </div>
        {mode === 'real' && (
          <div className="seg" role="group" aria-label="Prompt format" data-testid="format-switch">
            <button type="button" data-testid="format-chat" className={`seg-opt${chatFmt ? ' is-view' : ''}`} aria-pressed={chatFmt} onClick={() => setChatFmt(true)}>Chat</button>
            <button type="button" data-testid="format-text" className={`seg-opt${chatFmt ? '' : ' is-view'}`} aria-pressed={!chatFmt} onClick={() => setChatFmt(false)}>Continue text</button>
            <Term id="chat-template" variant="icon" label="About the chat template" testId="format-info"><Icon name="info" /></Term>
          </div>
        )}
        <span className="chip">
          <label>Temperature <input data-testid="temp-input" type="text" inputMode="decimal" autoComplete="off" value={tempText}
            onChange={(e) => {
              const text = normaliseDecimal(e.target.value)
              if (text === null) return
              setTempText(text)
              const v = decimalValue(text, 0, 2) // a number beyond the limits takes effect as the limit
              if (v !== null) setTemperature(v)
            }}
            // on leaving, the field shows the value in effect: "9" becomes "2", an empty field its last value
            onBlur={() => setTempText(String(temperature))}
            onKeyDown={(e) => {
              if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
              e.preventDefault()
              const v = stepDecimal(temperature, e.key === 'ArrowUp' ? 0.1 : -0.1, 0, 2)
              setTemperature(v)
              setTempText(String(v))
            }} /></label>
          <Term id="generation-settings" section="temperature" variant="icon" label="About temperature" testId="setting-info"
            tip="0 = always the top token; higher = more varied."><Icon name="info" /></Term>
        </span>
        <span className="chip">
          <label>Top-k <input data-testid="topk-input" type="number" min="1" max="10" value={topK}
            onChange={(e) => setTopK(clampOrKeep(e.target.value, 1, 10, topK))} /></label>
          <Term id="generation-settings" section="top-k" variant="icon" label="About top-k" testId="setting-info"
            tip="Only the k best candidates can be drawn."><Icon name="info" /></Term>
        </span>
        <span className="chip">
          <label>Max tokens <input data-testid="maxtok-input" type="number" min="1" max="100" value={maxNewTokens}
            onChange={(e) => setMaxNewTokens(clampOrKeep(e.target.value, 1, 100, maxNewTokens))} /></label>
          <Term id="generation-settings" section="max-new-tokens" variant="icon" label="About max new tokens" testId="setting-info"
            tip="The run stops here at the latest."><Icon name="info" /></Term>
        </span>
        {mode === 'real' && (
          <span className="chip">
            <label>Seed <input className="seed-input" data-testid="seed-input" inputMode="numeric" placeholder="random" value={seedText}
              onChange={(e) => { const v = parseSeed(e.target.value); if (v !== undefined) setSeedText(v === null ? '' : String(v)) }} /></label>
            <Term id="generation-settings" section="seed" variant="icon" label="About the seed" testId="setting-info"
              tip="Same seed, prompt and settings give the same text."><Icon name="info" /></Term>
          </span>
        )}
        <span className="toggles">
          {chatOn && (
            <button type="button" data-testid="toggle-system" className="btn btn-quiet" aria-expanded={systemOpen} aria-controls="system-row"
              onClick={() => setSystemToggled(!systemOpen)}>
              {systemEdited ? 'System prompt · edited' : 'System prompt'}<Icon name="chevron" />
            </button>
          )}
          {hasExamples && (
            <button type="button" data-testid="toggle-examples" className="btn btn-quiet" aria-expanded={examplesOpen} aria-controls="examples-row"
              onClick={() => setExamplesToggled(!examplesOpen)}>Examples<Icon name="chevron" /></button>
          )}
        </span>
      </div>
      {chatOn && systemOpen && (
        <div className="sub-row system-row" id="system-row" data-testid="system-row">
          <label htmlFor="system-input">System prompt</label>
          <input id="system-input" data-testid="system-input" value={system} onChange={(e) => setSystem(e.target.value)} />
          {systemEdited && (
            <button type="button" data-testid="system-reset" className="btn btn-quiet"
              onClick={() => setSystem(DEFAULT_SYSTEM_PROMPT)}>Reset to default</button>
          )}
        </div>
      )}
      {hasExamples && examplesOpen && (
        <div className="sub-row example-chip-row" id="examples-row" data-testid="examples-row">
          <span className="example-chip-label">Try:</span>
          {examples.map((ex) => (
            <button key={ex.id} data-testid="example-chip" className="example-chip pill" title={ex.hint}
              disabled={busy || generating}
              onClick={() => {
                setPrompt(ex.prompt)
                const params = { temperature, topK, maxNewTokens }
                if (ex.format === 'chat') {
                  setChatFmt(true); setSystem(DEFAULT_SYSTEM_PROMPT)
                  generate(ex.prompt, params, { kind: 'chat', system: DEFAULT_SYSTEM_PROMPT })
                } else {
                  setChatFmt(false)
                  generate(ex.prompt, params, TEXT_FORMAT)
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

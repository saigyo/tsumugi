import { useEffect, useRef, useState } from 'react'
import { Controls } from './app/Controls'
import { ModelStatus } from './app/ModelStatus'
import { PromptBar } from './app/PromptBar'
import { SimulatedEngine } from './engine/simulated/SimulatedEngine'
import { fallbackTokenizer, loadTokenizer, type Tokenizer } from './engine/tokenizer'
import { TransformersEngine } from './engine/transformers/TransformersEngine'
import type { PipelineEngine, ProgressInfo, RunHandle } from './engine/types'
import { usePlaybackTicker, usePlayerStore } from './player/store'
import { useTraceStore } from './trace/store'
import type { GenParams, Mode } from './trace/types'
import { DetailPanel } from './viz/DetailPanel'
import { PipelineBand } from './viz/PipelineBand'
import { TokenStream } from './viz/TokenStream'

export default function App() {
  usePlaybackTicker()
  const events = useTraceStore((s) => s.events)
  const cursor = usePlayerStore((s) => s.cursor)
  const [mode, setMode] = useState<Mode>('sim')
  const [busy, setBusy] = useState(false)
  const tokenizerRef = useRef<Tokenizer>(fallbackTokenizer())
  const runRef = useRef<RunHandle | null>(null)
  const [progress, setProgress] = useState<ProgressInfo | null>(null)
  const [modelError, setModelError] = useState<string | null>(null)
  const realEngineRef = useRef<TransformersEngine | null>(null)
  const [device, setDevice] = useState<'webgpu' | 'wasm' | null>(null)
  const [realReady, setRealReady] = useState(false)

  useEffect(() => {
    let live = true
    loadTokenizer().then((t) => { if (live) tokenizerRef.current = t })
    return () => { live = false }
  }, [])

  // trace growth → player
  useEffect(() => {
    usePlayerStore.getState().dispatch({ type: 'traceGrew', length: events.length })
  }, [events.length])

  const handleModeChange = async (m: Mode) => {
    setMode(m)
    setModelError(null)
    if (m === 'real' && !realEngineRef.current) {
      setRealReady(false)
      try {
        const engine = new TransformersEngine()
        await engine.prepare((p) => setProgress(p))
        realEngineRef.current = engine
        setDevice(engine.device)
        setRealReady(true)
      } catch (err) {
        setModelError(err instanceof Error ? err.message : String(err))
        setMode('sim')
        setRealReady(false)
      } finally {
        setProgress(null)
      }
    } else if (m === 'real' && realEngineRef.current) {
      setRealReady(true)
    }
  }

  const handleGenerate = async (prompt: string, params: GenParams) => {
    runRef.current?.abort()
    await runRef.current?.done
    useTraceStore.getState().clear()
    usePlayerStore.getState().dispatch({ type: 'reset' })
    const engine: PipelineEngine =
      mode === 'real' && realEngineRef.current
        ? realEngineRef.current
        : new SimulatedEngine(tokenizerRef.current)
    setBusy(true)
    try {
      const handle = engine.run(prompt, params, (e) => useTraceStore.getState().append(e))
      runRef.current = handle
      handle.done.finally(() => setBusy(false))
    } catch (err) {
      setBusy(false)
      setModelError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="app">
      <h1>LLM Pipeline Visualizer</h1>
      <ModelStatus progress={progress} device={mode === 'real' ? device : null} error={modelError}
        onFallback={() => { setModelError(null); setMode('sim') }} />
      <PromptBar mode={mode} onModeChange={handleModeChange} onGenerate={handleGenerate}
        busy={busy || (mode === 'real' && !realReady)} />
      <TokenStream events={events} cursor={cursor} />
      <PipelineBand events={events} cursor={cursor} />
      <DetailPanel events={events} cursor={cursor} mode={mode} />
      <Controls />
    </div>
  )
}

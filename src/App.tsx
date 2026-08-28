import { useEffect, useRef, useState } from 'react'
import { Controls } from './app/Controls'
import { ModelStatus } from './app/ModelStatus'
import { PromptBar } from './app/PromptBar'
import { usePins } from './app/usePins'
import { CURATED_EXAMPLES } from './engine/simulated/examples'
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
  const tokenizerRef = useRef<Tokenizer>(fallbackTokenizer())
  const runRef = useRef<RunHandle | null>(null)
  const [progress, setProgress] = useState<ProgressInfo | null>(null)
  const [modelError, setModelError] = useState<string | null>(null)
  const realEngineRef = useRef<TransformersEngine | null>(null)
  const preparingRef = useRef<{ engine: TransformersEngine; promise: Promise<void> } | null>(null)
  const [device, setDevice] = useState<'webgpu' | 'wasm' | null>(null)
  const [realReady, setRealReady] = useState(false)
  const [attn, setAttn] = useState(false)
  const { pins, note: pinNote, pin: handlePin, reset: resetPins } = usePins((layer, head) => {
    const engine = realEngineRef.current
    return engine
      ? engine.fetchHead(layer, head)
      : Promise.resolve({ layer, head, matrix: [], label: null, score: null })
  })

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
      let prepping = preparingRef.current
      if (!prepping) {
        const engine = new TransformersEngine()
        prepping = { engine, promise: engine.prepare((p) => setProgress(p)) }
        preparingRef.current = prepping
      }
      try {
        await prepping.promise
        realEngineRef.current = prepping.engine
        setDevice(prepping.engine.device)
        setAttn(prepping.engine.attentions ?? false)
        setRealReady(true)
      } catch (err) {
        setModelError(err instanceof Error ? err.message : String(err))
        setMode('sim')
        setRealReady(false)
      } finally {
        if (preparingRef.current === prepping) preparingRef.current = null
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
    resetPins()
    const engine: PipelineEngine =
      mode === 'real' && realEngineRef.current
        ? realEngineRef.current
        : new SimulatedEngine(tokenizerRef.current)
    try {
      const handle = engine.run(prompt, params, (e) => useTraceStore.getState().append(e))
      runRef.current = handle
    } catch (err) {
      setModelError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="app">
      <h1><span className="app-mark">紬</span> Tsumugi <span className="app-subtitle">LLM Pipeline Visualizer</span></h1>
      <PromptBar mode={mode} onModeChange={handleModeChange} onGenerate={handleGenerate}
        busy={mode === 'real' && !realReady} examples={CURATED_EXAMPLES}
        status={<ModelStatus progress={progress} device={mode === 'real' ? device : null} error={modelError}
          attentions={mode === 'real' && attn}
          onFallback={() => { setModelError(null); setMode('sim') }} />} />
      <TokenStream events={events} cursor={cursor} />
      <PipelineBand events={events} cursor={cursor} onStageClick={(index) => {
        usePlayerStore.getState().dispatch({ type: 'seek', index })
        usePlayerStore.getState().dispatch({ type: 'pause' })
      }} />
      <DetailPanel events={events} cursor={cursor} mode={mode}
        pinnedHeads={pins} onPin={handlePin} pinNote={pinNote} />
      <Controls />
    </div>
  )
}

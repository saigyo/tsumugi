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
import { RunShelf } from './app/RunShelf'
import { archiveImport, archiveRemove, archiveSeal, archiveTogglePin, initArchive } from './app/runArchive'
import { parseRunFile, serializeRun } from './app/runFiles'
import { createIndexedDbStorage } from './app/runStorage'
import { useRunsStore } from './app/runsStore'
import { CompareView } from './viz/compare/CompareView'

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
  const records = useRunsStore((s) => s.records)
  const activeId = useRunsStore((s) => s.activeId)
  const persistFailed = useRunsStore((s) => s.persistFailed)
  const [compare, setCompare] = useState<{ aId: string; bId: string } | null>(null)
  const [compareArmed, setCompareArmed] = useState(false)
  const [importError, setImportError] = useState<string | null>(null)
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

  useEffect(() => { void initArchive(createIndexedDbStorage()) }, [])
  useEffect(() => { resetPins() }, [activeId, resetPins])

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
    setCompare(null)
    setCompareArmed(false)
    const engine: PipelineEngine =
      mode === 'real' && realEngineRef.current
        ? realEngineRef.current
        : new SimulatedEngine(tokenizerRef.current)
    try {
      const handle = engine.run(prompt, params, (e) => {
        useTraceStore.getState().append(e)
        if (e.type === 'run-end') {
          // seal metadata comes from the trace itself — run-start carries
          // prompt/params/mode/modelId; the archive is trace-derived by design
          const events = useTraceStore.getState().events
          const start = events.find((x) => x.type === 'run-start')
          if (start && start.type === 'run-start') {
            archiveSeal({
              prompt: start.prompt, params: start.params, mode: start.mode,
              ...(start.mode === 'real' ? { modelId: start.modelId } : {}),
              endedAt: Date.now(), reason: e.reason,
            }, events)
          }
        }
      })
      runRef.current = handle
    } catch (err) {
      setModelError(err instanceof Error ? err.message : String(err))
    }
  }

  const handleActivate = async (id: string) => {
    runRef.current?.abort()
    await runRef.current?.done
    const record = useRunsStore.getState().records.find((r) => r.id === id)
    if (!record) return
    useRunsStore.getState().setActive(id)
    useTraceStore.getState().load(record.events)
    const dispatch = usePlayerStore.getState().dispatch
    dispatch({ type: 'traceGrew', length: record.events.length })
    dispatch({ type: 'seek', index: record.events.length - 1 })
    dispatch({ type: 'pause' })
  }

  const handleRemove = (id: string) => {
    archiveRemove(id)
    setCompare((c) => (c && (c.aId === id || c.bId === id) ? null : c))
  }

  const handleExport = (id: string) => {
    const record = useRunsStore.getState().records.find((r) => r.id === id)
    if (!record) return
    const { filename, json } = serializeRun(record)
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = filename
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const handleImportFile = (file: File) => {
    void file.text().then((text) => {
      const parsed = parseRunFile(text)
      if (parsed.ok) { setImportError(null); archiveImport({ meta: parsed.meta, events: parsed.events }) }
      else setImportError(parsed.error)
    })
  }

  return (
    <div className="app">
      <h1><span className="app-mark">紬</span> Tsumugi <span className="app-subtitle">LLM Pipeline Visualizer</span></h1>
      <PromptBar mode={mode} onModeChange={handleModeChange} onGenerate={handleGenerate}
        busy={mode === 'real' && !realReady} examples={CURATED_EXAMPLES}
        status={<ModelStatus progress={progress} device={mode === 'real' ? device : null} error={modelError}
          attentions={mode === 'real' && attn}
          onFallback={() => { setModelError(null); setMode('sim') }} />} />
      <RunShelf records={records} activeId={activeId} compare={compare} armed={compareArmed}
        persistFailed={persistFailed} importError={importError}
        onActivate={(id) => { void handleActivate(id) }}
        onSelectCompareB={(id) => {
          setCompareArmed(false)
          setCompare((c) => c ? { ...c, bId: id } : activeId ? { aId: activeId, bId: id } : null)
        }}
        onArmCompare={() => setCompareArmed(true)}
        onExitCompare={() => { setCompare(null); setCompareArmed(false) }}
        onTogglePin={archiveTogglePin} onRemove={handleRemove}
        onExport={handleExport} onImportFile={handleImportFile} />
      {(() => {
        const cmpA = compare && records.find((r) => r.id === compare.aId)
        const cmpB = compare && records.find((r) => r.id === compare.bId)
        if (cmpA && cmpB) return <CompareView a={cmpA} b={cmpB} />
        return (
          <>
            <TokenStream events={events} cursor={cursor} />
            <PipelineBand events={events} cursor={cursor} onStageClick={(index) => {
              usePlayerStore.getState().dispatch({ type: 'seek', index })
              usePlayerStore.getState().dispatch({ type: 'pause' })
            }} />
            <Controls />
            <DetailPanel events={events} cursor={cursor} mode={mode}
              pinnedHeads={pins} onPin={handlePin} pinNote={pinNote} />
          </>
        )
      })()}
    </div>
  )
}

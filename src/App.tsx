import { useEffect, useRef, useState } from 'react'
import { Controls } from './app/Controls'
import { PromptBar } from './app/PromptBar'
import { SimulatedEngine } from './engine/simulated/SimulatedEngine'
import { fallbackTokenizer, loadTokenizer, type Tokenizer } from './engine/tokenizer'
import type { PipelineEngine, RunHandle } from './engine/types'
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

  useEffect(() => {
    let live = true
    loadTokenizer().then((t) => { if (live) tokenizerRef.current = t })
    return () => { live = false }
  }, [])

  // trace growth → player
  useEffect(() => {
    usePlayerStore.getState().dispatch({ type: 'traceGrew', length: events.length })
  }, [events.length])

  const handleGenerate = async (prompt: string, params: GenParams) => {
    runRef.current?.abort()
    await runRef.current?.done
    useTraceStore.getState().clear()
    usePlayerStore.getState().dispatch({ type: 'reset' })
    const engine: PipelineEngine = new SimulatedEngine(tokenizerRef.current)  // real engine wired in Task 18
    setBusy(true)
    const handle = engine.run(prompt, params, (e) => useTraceStore.getState().append(e))
    runRef.current = handle
    handle.done.finally(() => setBusy(false))
  }

  return (
    <div className="app">
      <h1>LLM Pipeline Visualizer</h1>
      <PromptBar mode={mode} onModeChange={setMode} onGenerate={handleGenerate} busy={busy} />
      <TokenStream events={events} cursor={cursor} />
      <PipelineBand events={events} cursor={cursor} />
      <DetailPanel events={events} cursor={cursor} mode={mode} />
      <Controls />
    </div>
  )
}

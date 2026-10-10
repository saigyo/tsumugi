import type { GlossaryEntry } from '../types'

export const modes: GlossaryEntry = {
  id: 'modes',
  title: 'Simulated and real mode',
  summary: 'Simulated mode shows scripted, illustrative data; real mode runs the SmolLM2-135M model in your browser.',
  stage: 'general',
  related: ['how-to-read', 'attention'],
  body: () => (
    <>
      <p><strong>Simulated</strong> needs no download and starts at once. A small program produces tokens, scores
        and attention patterns that illustrate each stage. The curated examples follow a script.</p>
      <p><strong>Real</strong> downloads SmolLM2-135M-Instruct once (about 250 MB, quantized ONNX) and runs it in
        the browser on WebGPU, or on WebAssembly where WebGPU is missing. Nothing leaves your machine. Every number
        shown comes from the model, including attention weights and embedding rows.</p>
      <p className="gl-aside">Where real mode cannot show a value, the card says so (for example “schematic”).</p>
    </>
  ),
  inRun: (run) => (
    <p>This run used {run.mode === 'real' ? `the real model (${run.modelId})` : 'the simulated model'}.</p>
  ),
}

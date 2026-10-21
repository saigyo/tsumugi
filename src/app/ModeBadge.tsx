import type { Mode } from '../trace/types'

export function ModeBadge({ mode, loading, device, attentions }: {
  mode: Mode; loading: boolean; device: 'webgpu' | 'wasm' | null; attentions: boolean
}) {
  const text = mode === 'sim' ? 'Simulated' : loading || !device ? 'Loading…' : `Real model · ${device}${attentions ? ' · attn' : ''}`
  return <span data-testid="mode-badge" className={`mode-badge mode-badge-${mode === 'sim' ? 'sim' : 'real'}`}>{text}</span>
}

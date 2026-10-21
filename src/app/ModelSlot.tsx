import type { ProgressInfo } from '../engine/types'
import type { Mode } from '../trace/types'

// The one place for the model's status: the banner while simulated, the
// download progress while loading, a load error with its fallback.
export function ModelSlot({ mode, loading, fromCache, progress, error, bannerClosed, onLoad, onClose, onFallback }: {
  mode: Mode; loading: boolean; fromCache: boolean; progress: ProgressInfo | null; error: string | null
  bannerClosed: boolean; onLoad(): void; onClose(): void; onFallback(): void
}) {
  if (error) return (
    <div role="status" data-testid="model-error" className="model-slot model-error">
      Real model unavailable: {error}
      <button data-testid="btn-fallback" onClick={onFallback}>Continue in Simulated mode</button>
    </div>
  )
  if (loading) {
    const pct = progress && progress.total > 0 ? Math.round((progress.loaded / progress.total) * 100) : null
    return (
      <div role="status" data-testid="model-progress" className="model-slot model-progress">
        {fromCache ? 'Loading the real model from the browser’s cache…'
          : progress ? `Loading the real model: ${progress.file} · ${pct ?? 0} %` : 'Loading the real model…'}
        {pct !== null && <progress aria-label="Download progress" data-testid="model-progress-bar" max={100} value={pct} />}
      </div>
    )
  }
  if (mode !== 'sim' || bannerClosed) return null
  return (
    <div data-testid="sim-banner" className="model-slot sim-banner">
      <p><strong>You are watching a simulation.</strong> Its numbers are made up to show the shape of the pipeline.
        Load the real model to see what SmolLM2 actually computes: real attention, the MLP inside each layer, and
        answers in Chat mode. 250 MB, downloaded once, runs entirely in your browser.</p>
      <button type="button" data-testid="banner-load" className="load-real" onClick={onLoad}>Load the real model</button>
      <button type="button" className="sim-banner-close" aria-label="Close" onClick={onClose}>×</button>
    </div>
  )
}

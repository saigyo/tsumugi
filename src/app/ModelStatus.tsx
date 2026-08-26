import type { ProgressInfo } from '../engine/types'

interface Props {
  progress: ProgressInfo | null
  device: 'webgpu' | 'wasm' | null
  error: string | null
  onFallback(): void
}

export function ModelStatus({ progress, device, error, onFallback }: Props) {
  if (error) return (
    <div data-testid="model-error" className="model-error">
      Real model unavailable: {error}
      <button data-testid="btn-fallback" onClick={onFallback}>Continue in Simulated mode</button>
    </div>
  )
  if (progress) return (
    <div data-testid="model-progress" className="model-progress">
      Downloading {progress.file}: {progress.total > 0 ? Math.round((progress.loaded / progress.total) * 100) : 0}%
    </div>
  )
  if (device) return <span data-testid="device-chip" className="device-chip">{device}</span>
  return null
}

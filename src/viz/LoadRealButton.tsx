import { useRealModel } from './realModelRequest'

export function LoadRealButton({ variant = 'button' }: { variant?: 'button' | 'link' }) {
  const request = useRealModel((s) => s.request)
  const loading = useRealModel((s) => s.loading)
  return (
    <button type="button" data-testid="load-real" className={variant === 'link' ? 'link-button' : 'load-real'}
      disabled={loading} onClick={request}>Load the real model</button>
  )
}

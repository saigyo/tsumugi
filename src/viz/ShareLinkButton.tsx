import { useEffect, useRef, useState } from 'react'
import { buildShareLink, type ShareSettings } from '../app/shareLink'
import { useRunRequest } from './runRequest'
import { Icon } from '../ui/Icon'

export type ShareFeedback = { kind: 'copied' } | { kind: 'failed'; link: string }

export function ShareLinkButton({ settings, compact, onFeedback }: {
  settings: ShareSettings
  compact?: boolean
  onFeedback?: (f: ShareFeedback | null) => void
}) {
  const busy = useRunRequest((s) => s.busy)
  const [local, setLocal] = useState<ShareFeedback | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])

  const report = (f: ShareFeedback | null) => (onFeedback ?? setLocal)(f)
  const copy = async () => {
    const link = buildShareLink(location.origin + location.pathname, settings)
    clearTimeout(timer.current)
    try {
      await navigator.clipboard.writeText(link)
      report({ kind: 'copied' })
      timer.current = setTimeout(() => report(null), 2000)
    } catch {
      report({ kind: 'failed', link })
    }
  }

  return (
    <>
      {compact
        ? <button type="button" data-testid="btn-chip-share" className="run-chip-action" title="copy share link"
            aria-label="Copy link" disabled={busy} onClick={copy}><Icon name="link" /></button>
        : <button type="button" className="btn" disabled={busy} onClick={copy}>Copy link</button>}
      {!onFeedback && local?.kind === 'copied' && <span role="status" className="share-feedback">Link copied</span>}
      {!onFeedback && local?.kind === 'failed' && (
        <input readOnly data-testid="share-link-text" className="share-link-text" value={local.link}
          aria-label="Share link" onFocus={(e) => e.currentTarget.select()} />
      )}
    </>
  )
}

import { useEffect, useState } from 'react'
import type { TraceEvent } from '../trace/types'
import { LAYER_SECTIONS } from './layersSections'
import { useMlpLayerControl } from './mlpLayerSelection'
import type { VotedCandidate } from './selectors'

// The Layers detail's sticky section bar (#17 spec, section 2): jump links,
// the section on screen highlighted, and the shared MLP layer stepper.
// Publishes an element's height as a CSS custom property on <html> (the bar
// sticks below the playback bar, and section scroll margins clear both bars).
function usePublishHeight(el: Element | null, name: string) {
  useEffect(() => {
    if (!el || typeof ResizeObserver === 'undefined') return
    const root = document.documentElement
    const ro = new ResizeObserver(() => root.style.setProperty(name, `${el.getBoundingClientRect().height}px`))
    ro.observe(el)
    return () => { ro.disconnect(); root.style.removeProperty(name) }
  }, [el, name])
}

export function LayersNav({ candidates, run, showStepper }: { candidates?: VotedCandidate[]; run?: TraceEvent; showStepper: boolean }) {
  const [barEl, setBarEl] = useState<HTMLElement | null>(null)
  const [current, setCurrent] = useState(LAYER_SECTIONS[0].id)
  const { layer, layers, step } = useMlpLayerControl(showStepper ? candidates : undefined, run)
  usePublishHeight(barEl, '--nav-h')
  usePublishHeight(barEl && document.querySelector('.controls'), '--controls-h')
  useEffect(() => {
    let frame = 0
    const update = () => {
      frame = 0
      // a section scrolled to (scroll-margin: bar + 6 px) must count as reached
      const line = (barEl?.getBoundingClientRect().bottom ?? 0) + 8
      let id = LAYER_SECTIONS[0].id
      for (const s of LAYER_SECTIONS) {
        const el = document.getElementById(s.id)
        if (el && el.getBoundingClientRect().top <= line) id = s.id
      }
      setCurrent(id)
    }
    const onScroll = () => { if (!frame) frame = requestAnimationFrame(update) }
    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => { window.removeEventListener('scroll', onScroll); if (frame) cancelAnimationFrame(frame) }
  }, [barEl])
  return (
    <nav ref={setBarEl} data-testid="layers-nav" className="layers-nav" aria-label="Layers sections">
      <span className="layers-nav-title">Jump to</span>
      <div className="layers-nav-links">
        {LAYER_SECTIONS.map((s) => (
          <a key={s.id} href={`#${s.id}`} data-testid="nav-link" data-section={s.id} className="layers-nav-link"
            aria-current={s.id === current ? 'true' : undefined}
            onClick={(e) => { e.preventDefault(); document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }}>
            {s.label}</a>
        ))}
      </div>
      {showStepper && layer !== null && (
        <span data-testid="nav-layer" className="layers-nav-layer">MLP layer {layer}
          <button type="button" data-testid="nav-prev" aria-label="Previous MLP layer" disabled={layer <= 0} onClick={() => step(-1)}>◀</button>
          <button type="button" data-testid="nav-next" aria-label="Next MLP layer" disabled={layer >= layers - 1} onClick={() => step(1)}>▶</button>
        </span>
      )}
    </nav>
  )
}

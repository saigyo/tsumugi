import type { ReactNode } from 'react'

// A part of an entry body that term links and callout chips can scroll to.
export function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section data-gl-section={id} className="gl-section">
      <h4>{title}</h4>
      {children}
    </section>
  )
}

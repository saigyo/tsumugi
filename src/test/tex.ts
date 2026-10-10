// The LaTeX source a <Formula> was rendered from (temml's annotate option
// embeds it), plus whether it rendered as MathML at all.
export function texOf(el: Element): string | null {
  if (!el.querySelector('math')) return null
  return el.querySelector('annotation')?.textContent ?? null
}

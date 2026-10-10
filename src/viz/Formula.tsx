import { texToMathML } from './tex'

// A formula written in LaTeX, rendered as native MathML. Invalid LaTeX shows
// its source, marked, rather than breaking the card (the registry and card
// tests fail on any such marker).
export function Formula({ tex, inline = false, testId, className }: {
  tex: string
  inline?: boolean
  testId?: string
  className?: string
}) {
  const Tag = inline ? 'span' : 'div'
  let html: string
  try {
    html = texToMathML(tex, inline)
  } catch (err) {
    console.error('Formula: invalid LaTeX', tex, err)
    return <Tag data-testid={testId} data-formula-error="" className="formula formula-error"><code>{tex}</code></Tag>
  }
  return <Tag data-testid={testId} className={`formula${inline ? ' formula-inline' : ''}${className ? ` ${className}` : ''}`}
    dangerouslySetInnerHTML={{ __html: html }} />
}

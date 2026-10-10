import temml from 'temml'
import { thousands } from './selectors'

// LaTeX → MathML for the Formula component. Kept apart from the component so
// that file exports only components. Results are cached: the formulas are a
// fixed set plus a few run-dependent numbers.
const cache = new Map<string, string>()

// Throws temml's ParseError on invalid LaTeX; the Formula component catches it.
export function texToMathML(tex: string, inline = false): string {
  const key = `${inline ? 'i' : 'd'}${tex}`
  let html = cache.get(key)
  if (html === undefined) {
    html = temml.renderToString(tex, { displayMode: !inline, annotate: true, throwOnError: true })
    cache.set(key, html)
  }
  return html
}

// 49152 → '49\,152': thousands grouped with a thin space, as in the cards' text.
export const texNumber = (n: number) => thousands(n).replace(/ /g, '\\,')

// The cards' formula lines, with the run's own sizes. Rendered inline (left
// aligned like the rest of the card) but in display style, so fractions keep
// their full size.
export const logitsTex = (dims: number, vocab: number) =>
  String.raw`\displaystyle \text{logits} = x_{\text{last}} \cdot E^{\top} \qquad [1 \times ${dims}] \cdot [${dims} \times ${texNumber(vocab)}] \to [${texNumber(vocab)}]`

export const softmaxTex = (shown: number, vocab?: number) =>
  String.raw`\displaystyle p = \frac{\exp(\text{logit}/T)}{\sum \exp(\text{logit}/T)} \qquad \text{over the top-}k\text{: } ${shown} \text{ of } ${vocab ? texNumber(vocab) : '?'} \text{ scores}`

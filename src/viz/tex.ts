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

// The cards' formula lines, with the run's own sizes: the formula on the first
// row, the shapes (or the top-k note) on the second, so the line never gets
// wider than a phone. Rendered inline (left aligned like the rest of the card)
// but in display style, so fractions keep their full size.
export const logitsTex = (dims: number, vocab: number) =>
  String.raw`\displaystyle \begin{aligned} &\text{logits} = \operatorname{norm}(x_{\text{last}}) \cdot E^{\top} \\ &[1 \times ${dims}] \cdot [${dims} \times ${texNumber(vocab)}] \to [${texNumber(vocab)}] \end{aligned}`

export const softmaxTex = (shown: number, vocab?: number) =>
  String.raw`\displaystyle \begin{aligned} &p_i = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)} \\ &\text{over the top-}k\text{: } ${shown} \text{ of } ${vocab ? texNumber(vocab) : '?'} \text{ scores} \end{aligned}`

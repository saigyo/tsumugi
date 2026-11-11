import { fixed, percent } from '../ui/format'

// Copy for the Softmax card, computed from the run's own temperature so the
// paragraph, the per-bar derivations and the glossary's Softmax entry agree.
// Kept out of the component files so those export only components
// (fast-refresh lint rule).

// How much more probable a candidate d points ahead is, at temperature T:
// e^(d / T). The familiar "2.7× per point" holds only at T = 1.
export const leadRatio = (points: number, T: number) => fixed(Math.exp(points / T), 1)

// One candidate's share, derived on its own numbers: lead below the top score,
// the relative weight exp(−lead / T), and the resulting percentage.
export function shareDerivation(logit: number, top: number, T: number, prob: number): string {
  const share = percent(prob)
  if (T === 0) return logit === top ? `T = 0: the top score takes everything → ${share}` : `T = 0: not the top score → ${share}`
  if (logit === top) return `the top score → weight 1.00 → ${share}`
  const lead = top - logit
  const w = Math.exp(-lead / T)
  return `${fixed(lead, 1)} below the top score → exp(−${fixed(lead, 1)} / ${T}) = ${fixed(w, 2)} of the leader’s weight → ${share}`
}

export function softmaxExplainer(T: number, shown: number, vocab: string): string {
  if (T === 0)
    return 'At T = 0 there is no draw: the top score takes 100% and the rest 0% — greedy decoding. (Dividing by zero would blow the exponentials up, so the sampler simply picks the maximum.)'
  return `Each score is divided by the temperature T = ${T}, then exponentiated; each result is divided by the sum of all of them, so the shares add up to 100%. Exponentiating turns differences in score into ratios of e^(difference / T): at T = ${T} a lead of 1 point is a ${leadRatio(1, T)}× lead in probability, 2 points is ${leadRatio(2, T)}×. A lower T stretches the differences and concentrates probability on the leader; a higher T flattens them. The sum runs over the top-k candidates only, so these are shares among the ${shown} shown, not of all ${vocab} tokens. Hover a bar for its own numbers.`
}

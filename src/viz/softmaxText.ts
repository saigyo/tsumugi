// How much more probable a candidate d points ahead is, at temperature T:
// e^(d / T). The familiar "2.7× per point" holds only at T = 1; the card's
// copy computes it from the run's actual T so it agrees with the per-bar
// derivations. Kept out of the component file so that file exports only
// components (fast-refresh lint rule).
export const leadRatio = (points: number, T: number) => Math.exp(points / T).toFixed(1)

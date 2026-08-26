export function softmax(logits: number[], temperature: number): number[] {
  if (temperature === 0) {
    const best = logits.indexOf(Math.max(...logits))
    return logits.map((_, i) => (i === best ? 1 : 0))
  }
  const scaled = logits.map((l) => l / temperature)
  const max = Math.max(...scaled)
  const exps = scaled.map((l) => Math.exp(l - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

export function topK(data: ArrayLike<number>, k: number): Array<{ id: number; logit: number }> {
  const entries: Array<{ id: number; logit: number }> = []
  for (let i = 0; i < data.length; i++) entries.push({ id: i, logit: data[i] })
  return entries.sort((a, b) => b.logit - a.logit).slice(0, k)
}

export function sampleIndex(probs: number[], rand: () => number): number {
  const r = rand()
  let acc = 0
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i]
    if (r < acc) return i
  }
  return probs.length - 1
}

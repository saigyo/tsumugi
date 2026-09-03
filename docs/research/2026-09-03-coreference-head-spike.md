# Does SmolLM2-135M have a coreference head the app can detect?

## Question

Backlog #5: real mode labels heads as previous-token, attention-sink, induction or
distinctive, but the *coreference* role (the poster child of the simulated examples,
"it → cat") has no real-mode detector. Two things had to be true for one to be worth
building:

1. The model actually has a head whose pronoun rows attend to the pronoun's referent,
   consistently enough to clear the app's showcase threshold (0.3, see
   `src/engine/transformers/attentionStats.ts`).
2. The app can find that head **without knowing the antecedent** — at run time it
   only has the token strings and the attention matrices.

## TL;DR

Yes on both counts. **Layer 13, head 8** (0-based, as the app numbers them) is a clean
coreference head in `HuggingFaceTB/SmolLM2-135M-Instruct`: from a pronoun's row it puts
0.5–0.97 of its attention on the referent, including object antecedents (*rode it* →
horse, *graded them* → essays, *rebuilt its* → town), so it is not merely a
"subject noun" heuristic. Layer 15 head 7 is a weaker runner-up. An antecedent-blind
template — mean over pronoun rows of the largest weight on an earlier *word* token,
excluding column 0, column i−1 **and the diagonal** — ranks L13H8 first of all 270 heads
at ≈0.65. Three constraints fall out of the spike: the diagonal must be excluded
(otherwise four self-attention heads win), function-word columns must be excluded
(otherwise a fixed-column head parked on " on" wins in the browser, section 4.1), and
curated examples must not put the referent at position 0 (the attention sink inflates
every head's score there).
Recommendation: build #5 as a small addition to the existing detector module.

---

## 1. Setup

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct` in PyTorch (`transformers` 4.57,
  `torch` 2.13, eager attention, fp32), run from the `tools/export` uv environment.
  The ONNX export used by the app is validated against this model (`inputs-embeds`
  and attention parity checks in `tools/export`), so attention weights are the same
  up to quantisation.
- Tokenisation mirrors the app's worker: raw prompt, no chat template, no BOS
  (`add_special_tokens=False`).
- 30 layers × 9 query heads = 270 heads, indexed 0-based as in the app's grid.
- Scripts were throwaway (session scratchpad, not committed); section 6 has enough
  to reproduce.

## 2. Ground-truth pass: which head attends pronoun → antecedent?

Twelve prompts, each with one pronoun whose referent is unambiguous and a distractor
noun. Words were chosen to be single BPE tokens where possible. Per head, the
measurements from the pronoun's row were: weight on the antecedent (`ante`), on the
distractor (`dist`), on the previous token, on column 0 (sink), and whether the
antecedent is the row's argmax once column 0 and column i−1 are ignored (`hit`). A
baseline `base` = the mean weight that *other* rows after the antecedent put on the
antecedent column, so `lift = ante − base` measures how pronoun-specific the head is.

### 2.1 First attempt was confounded by the sink

Half the prompts started with the referent ("Mary told John that she…"), so the
antecedent was token 0, which every head treats as the attention sink. The top heads by
`ante` had `base` ≈ 0.4 and `sink` ≈ 0.9: they attend to column 0 from every row and
told us nothing about coreference. Fix: prefix those prompts with "Yesterday, " so no
referent sits at position 0.

### 2.2 Results with the referent off position 0

| layer | head | mean lift | mean ante | base | mean dist | hits |
|---|---|---|---|---|---|---|
| 15 | 7 | 0.32 | 0.38 | 0.06 | 0.12 | 10/12 |
| 9 | 2 | 0.27 | 0.30 | 0.03 | 0.04 | 7/12 |
| 9 | 4 | 0.27 | 0.35 | 0.08 | 0.11 | 8/12 |
| 4 | 8 | 0.25 | 0.28 | 0.03 | 0.08 | 9/12 |
| **13** | **8** | 0.19 | **0.50** | 0.31 | 0.14 | **10/12** |

L13H8 has the highest antecedent weight and the best antecedent−distractor margin
(0.35); L15H7 is the most pronoun-specific (lowest base). Per-prompt detail for L13H8
(`ante / dist`):

| prompt | ante | dist |
|---|---|---|
| The cat sat on the mat because **it** was tired. | 0.82 | 0.02 |
| The trophy did not fit in the box because **it** was too big. | 0.56 | 0.04 |
| Yesterday, Mary told John that **she** would arrive late. | 0.51 | 0.13 |
| Yesterday, John told Mary that **he** would arrive late. | 0.33 | 0.06 |
| The bird saw the snake and **it** flew away quickly. | 0.15 | 0.51 |
| Yesterday, Tom lost the keys, so **he** was late for work. | 0.75 | 0.00 |
| Yesterday, Anna gave the book to Peter because **she** had finished it. | 0.49 | 0.17 |
| The children played in the garden until **they** were tired. | 0.66 | 0.01 |
| The lamp stood on the desk and **it** was very bright. | 0.77 | 0.06 |
| Yesterday, Sarah called the doctor because **she** felt sick. | 0.35 | 0.18 |
| The dog barked at the mailman until **he** walked away. | 0.02 | 0.27 |
| Yesterday, Paul thanked Lisa after **she** fixed the car. | 0.57 | 0.29 |

The two misses are instructive: "the bird saw the snake and it flew" is genuinely
ambiguous (the head picked *snake*), and "mailman" splits into ` mail` + `man`; the
head put 0.41 on `man`, the second sub-token, which the scorer did not count.

## 3. Held-out pass: object antecedents and expletives

Seven new prompts to separate "coreference" from "attend to the sentence subject".
Top attention target of the pronoun row in L13H8:

| pronoun row | target | weight |
|---|---|---|
| …because she had finished **it** | book | 0.96 |
| The farmer bought a horse and rode **it** home | horse | 0.97 |
| Lisa found a coin and put **it** in her pocket | coin | 0.94 |
| …put it in **her** pocket | L / isa (both sub-tokens) | 0.51 + 0.21 |
| The boy threw the ball and the dog caught **it** | ball | 0.80 |
| After the storm, the town rebuilt **its** bridge | town | 0.87 |
| …after **she** had graded them | teacher | 0.76 |
| …after she had graded **them** | essays | 0.92 |
| **It** was raining, so… (expletive, row 0) | itself | 1.00 (no earlier token) |

Object antecedents are resolved as reliably as subjects, and the pronoun in a
sentence with two candidate nouns ("the boy … the dog caught it") goes to the right
one. L15H7 agrees on most of these but with lower weight (0.48–0.82) and fails on the
possessive *its*. Heads 9/2, 9/4 and 4/8 were mixed and are not candidates.

## 4. Can the app find it blind?

The detector cannot know the antecedent, so the template has to be defined purely
from the row. Candidate score per head:

> mean over pronoun rows *i* (i ≥ 2) of max<sub>j</sub> A[i][j], over columns *j* that
> are word tokens (alphabetic, length > 1), excluding j = 0 (sink), j = i−1
> (previous token) and j = i (self).

Pronoun list used: it, he, she, they, his, her, its, their, him, them. A prompt with
no pronoun rows yields `null`, like the induction score does when no token repeats.

Ranking over the 17 prompts of sections 2 and 3:

| rank | head | blind score |
|---|---|---|
| 1 | L13H8 | 0.65 |
| 2 | L16H5 | 0.49 |
| 3 | L15H7 | 0.42 |
| 4 | L14H8 | 0.41 |
| 5 | L3H8 | 0.39 |

A lift variant (peak weight minus other rows' weight on the same column) also ranks
L13H8 first (0.38), with L15H7 second (0.36), so the plain score suffices.

**The diagonal exclusion is essential.** Without j = i masked, the ranking is led by
L11H4, L29H1, L11H6 and L5H2 — heads whose pronoun rows attend to *themselves*
(0.6–0.9) — and L13H8 drops to second. Those are not coreference heads.

Per-prompt winners under the blind score are L13H8 on 8 of 17 prompts; on the others a
different head wins by a small margin (e.g. L9H3 on the "told … that she" pair, L16H5
on "Paul thanked Lisa after she"). The showcase hysteresis in
`selectShowcaseHeads` (incumbent keeps its seat unless beaten by ≥ 0.05) will damp
that as it does for the other roles.

### 4.1 Amendment from the in-app check: exclude function-word columns

Running the detector in the browser (WebGPU, quantised export) on the cat example
picked **L5H5 at 0.68** instead of L13H8. Its heatmap shows why: from " the" onward
every row parks most of its weight on the column " on" — a fixed-column head, not
coreference — and the pronoun row goes there too, so the plain template cannot tell
it apart. (L13H8 does resolve *it → cat* in the browser as well; pinned over the whole
run it scored 0.49.)

Two fixes were measured in PyTorch over the 17 prompts plus two more:

| variant | L13H8 on the cat prompt | winner on the cat prompt | prompts L13H8 wins |
|---|---|---|---|
| plain (section 4) | 0.82 | L13H8 | 10 / 19 |
| lift (peak minus other rows' weight on that column) | 0.20 | L9H2 0.45 | 4 / 19 |
| plain + function-word stoplist | 0.82 | L13H8 | 10 / 19 |

Lift fails on the example itself: L13H8 attends to " cat" from most rows of that
prompt (base ≈ 0.6), so subtracting the base erases the signal, and the per-prompt
lift winners scatter across heads. The stoplist keeps the plain score and simply
refuses articles, prepositions, conjunctions, auxiliaries and modals as candidate
columns; L5H5's " on" drops to ≈ 0. That is a linguistic prior, but a defensible one —
a referent is a content word — and every remaining winner points the pronoun row at a
content word the reader can judge. Shipped with the stoplist.

## 5. Recommendation

Proceed with #5 as a bounded feature, not a research project:

- **Detector.** Add `corefScore: number | null` to `HeadStats`, computed as in
  section 4, and a fourth `pick('coreference', …)` at the default 0.3 threshold in
  `selectShowcaseHeads`; add it to the candidate list in `resolveHeadLabel` so pinned
  heads can receive the label. Word-token test (with the function-word stoplist of
  section 4.1) and pronoun list live next to the induction target logic.
- **Honesty of the label.** The blind score says "pronoun rows point at one earlier
  word", not "the right word". Keep the existing hint text ("Follow the pronoun's row:
  it attends back to its antecedent") but the curated example should invite the reader
  to check *which* word it landed on, and the ambiguous-sentence case is a good
  teaching moment rather than a bug.
- **Curated real-mode example.** Do not start the prompt with the referent (sink
  confound). "The farmer bought a horse and rode it home." (0.97) or "The lamp stood on
  the desk and it was very bright." (0.77) both work; the simulated example "The cat
  sat on the mat because it was tired" transfers directly (0.82).
- **Sub-token referents.** Names or nouns that split into several BPE tokens spread
  the weight across their pieces; the score uses the max over columns, so it
  under-reads them slightly. Acceptable for a showcase; a per-word merge is not worth
  the complexity.

## 6. Reproduction sketch

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=torch.float32).eval()
PRONOUNS = {"it","he","she","they","his","her","its","their","him","them"}

ids = tok.encode("The farmer bought a horse and rode it home.", add_special_tokens=False)
tokens = [tok.decode([i]) for i in ids]
with torch.no_grad():
    att = model(torch.tensor([ids]), output_attentions=True).attentions  # 30 × (1, 9, S, S)
word = torch.tensor([t.strip().isalpha() and len(t.strip()) > 1 for t in tokens])
rows = [p for p, t in enumerate(tokens) if t.strip().lower() in PRONOUNS and p >= 2]
score = torch.zeros(30, 9)
for l in range(30):
    for p in rows:
        row = att[l][0, :, p, :].clone()
        row[:, 0] = 0; row[:, p - 1] = 0; row[:, p] = 0; row[:, ~word] = 0
        score[l] += row.max(dim=1).values
score /= max(len(rows), 1)
print(divmod(score.argmax().item(), 9), score.max().item())   # → (13, 8), ≈0.97
```

Run with `uv run --project tools/export python <script>`; the model is in the HF cache
after any export run.

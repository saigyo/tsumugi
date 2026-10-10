# In-app glossary — design

**Date:** 2026-10-10
**Status:** Approved design, pre-implementation
**Backlog:** #14 (do before #15, Opening the MLP)

## Purpose

The app has outgrown "read the README first". Visitors open the deployed
site without seeing the repository, and the cards use terms (residual
stream, edit, logit, temperature, …) that the README explains but the app
does not. The explanations the app does have are `title` tooltips on ⓘ
chips: short, unformatted, and invisible on touch screens and to keyboard
users.

This work adds one glossary of concept entries, reachable from the place
where a term is used and from a global "?" button, shown in a side drawer
that leaves the visualisation visible.

## Decisions already made

- **Audience: layered.** Every entry has a plain one-sentence summary for
  everyone (the hover text) and a body that goes technical — formulas and
  SmolLM2's real shapes (576 dimensions, 49 152 tokens, 30 layers) — for
  readers who keep going.
- **"In this run" block.** Entry text is fixed. An entry may add a block
  computed from the active run (for example the softmax lead ratios at the
  run's temperature). One entry per concept; no run-specific entries.
- **Entries are TSX.** One file per entry holds the text, the figures and
  the "In this run" function. TSX was chosen over Markdown because entries
  can embed the app's own visuals (`VectorStrip`, small SVGs) and every
  part of an entry stays in one source file.
- **First-visit hint, not an automatic drawer.** A dismissible note beside
  the "?" button points newcomers at "How to read Tsumugi".
- **Core set of 12 entries** in the first version (listed below). Further
  entries arrive with the features that need them.
- **Architecture A:** a small zustand store plus one drawer mounted in
  `App`. URL links to entries (`#glossary/<id>`) are deferred; they can be
  added later without changing this design.

## Terminology

- **Entry** — one glossary concept: title, summary, body, optional "In
  this run" block, related entries.
- **Term link** — an inline `<Term>` in a card that shows the summary on
  hover or focus and opens the entry on click.
- **Section** — a named part of an entry body that a link can scroll to.
- **Run summary** — the plain object `App` derives from the active trace
  and passes to "In this run" functions.
- **Index** — the drawer's home view listing all entries.

## Module layout

All new code lives in `src/glossary/`:

| File | Role |
|---|---|
| `types.ts` | `GLOSSARY_IDS`, `GlossaryId`, `GlossaryEntry`, `RunSummary`, `GlossaryStage` |
| `entries/*.tsx` | one file per entry, each exporting one `GlossaryEntry` object |
| `registry.ts` | ordered list of all entries (index order) |
| `store.ts` | zustand store: open entry, section, Back history, open/closed |
| `runSummary.ts` | builds a `RunSummary` from the trace at the viewed cycle |
| `Term.tsx` | inline term link (also used for callout chips) |
| `Section.tsx` | marks a linkable section inside an entry body |
| `GlossaryDrawer.tsx` | the drawer: entry view and index view |
| `HelpButton.tsx` | the "?" button and the first-visit hint |

Components and non-component exports stay in separate files (the oxlint
only-export-components rule). Entry files export only their entry object.

## Entry model

```ts
type GlossaryStage = 'general' | 'tokenize' | 'embed' | 'layers' | 'logits' | 'softmax' | 'sample'

const GLOSSARY_IDS = ['how-to-read', 'token', 'embedding', /* … */] as const
type GlossaryId = typeof GLOSSARY_IDS[number]

interface GlossaryEntry {
  id: GlossaryId            // stable key, kebab-case: 'residual-stream'
  title: string             // 'Residual stream'
  summary: string           // one plain sentence; hover text and index line
  stage: GlossaryStage      // groups the index in pipeline order
  related: GlossaryId[]     // shown as links under the body
  body: () => ReactNode     // TSX; may use <Term>, <Section>, app components
  inRun?: (run: RunSummary) => ReactNode | null
}
```

The ids are a plain constant list in `types.ts`, not derived from the
entry objects: entries refer to each other through `related` and
`<Term>`, and deriving the type from them would be circular. `<Term id>`,
`related` and chip targets are type-checked everywhere in the app; the
registry test checks that the registry holds exactly one entry per id.

### Run summary

Built in `App` from the active run's trace at the cycle the player is
showing; `null` when no run is loaded.

```ts
interface RunSummary {
  mode: Mode
  modelId: string
  params: GenParams                      // temperature, topK, maxNewTokens
  tokens: TokenInfo[]                    // the prompt's tokens
  cycle: number; cycles: number          // viewed cycle / cycles so far
  stream?: { seqLen: number; dims: number; layers: number }
  vocabSize?: number
  softmax?: { temperature: number; top: Array<{ text: string; logit?: number; prob: number }> }
}
```

The softmax entry reuses `leadRatio` and the other helpers in
`src/viz/softmaxText.ts`, so the drawer and the Softmax card compute the
same numbers.

### The 12 entries

| id | stage | covers |
|---|---|---|
| `how-to-read` | general | walk through the screen: prompt and mode, Generate, pipeline band and stage cards, detail panel, playback, run shelf and comparison |
| `token` | tokenize | text → token ids, the leading-space glyph ␣, special tokens in passing |
| `embedding` | embed | lookup of a row of E; sections `learned`, `position`, `tied` |
| `residual-stream` | layers | one [T × 576] matrix flowing through all layers |
| `edit` | layers | the vector a block adds; standard terms (block output, residual update) |
| `layer` | layers | attention then MLP, each adding an edit; 30 of them |
| `attention` | layers | weighted averages over positions; heatmap rows; the detected head roles (sink, previous-token, induction, coreference, distinctive) |
| `mlp` | layers | per-position widen → nonlinearity → narrow (576 → 1536 → 576) |
| `logit` | logits | score = x_last · E[token]; sections `tied`, `similar-rows`, `top-k` |
| `softmax` | softmax | scores → shares; sections `exp`, `temperature`, `top-k` |
| `sampling` | sample | greedy vs top-k draw, why output varies between runs |
| `modes` | general | simulated vs real model: what is computed, what is scripted |

"In this run" blocks, where the data exists: `how-to-read` (what the
reader is viewing: cycle, mode), `token` (the prompt's token count),
`residual-stream` and `edit` (the run's stream shape), `logit` (how many
scores, how many shown), `softmax` (lead ratios at the run's T, the
current top candidates), `sampling` (method and settings), `modes` (which
mode this run used).

## UI

### Term links

- Rendered as a `<button>` styled as running text with a dotted indigo
  underline; reachable with Tab, announced as a button.
- Hover or keyboard focus shows a small tooltip element (not a `title`
  attribute) with the summary and "click to read more".
- Click or Enter opens the entry in the drawer. Touch has no hover, so a
  tap opens the drawer directly.
- Optional `section` prop: the drawer scrolls to that section and
  highlights it briefly (respecting `prefers-reduced-motion`).
- An id unknown at runtime (only possible in legacy or untyped paths)
  renders the children as plain text.

### Drawer

- Fixed to the right edge, about 380 px wide, full height, card styling.
  Non-modal: the page stays visible, clickable and scrollable. It overlays
  rather than pushes the layout, so cards never reflow when it opens.
- Below 600 px viewport width it takes the full width.
- Entry view, top to bottom: Back link (only after navigating inside the
  drawer), title with stage chip and close button, body, "In this run"
  block (pale indigo fill, only when a run is loaded and `inRun` returns
  something), Related links, "All entries" link.
- Opening a term while the drawer is open replaces the entry and pushes
  the previous one onto the Back history. Closing clears the history.
- Escape or × closes. On open, focus moves to the entry heading; on close,
  it returns to the element that opened the drawer.
- If `inRun` throws, the block is skipped (an error boundary around it);
  the rest of the entry renders.

### "?" button and index

- A "?" button at the right end of the `h1` header line.
- It opens the drawer on the index: "How to read Tsumugi" first, then
  entries grouped by stage in the pipeline band's order (Tokenize,
  Embeddings, Layers, Logits, Softmax, Sampler), then general entries.
  Each row shows title and summary and opens the entry.

### First-visit hint

- On the first visit a small note beside the "?" button: "New here? Start
  with *How to read Tsumugi*", with a ×.
- Dismissed by ×, by following the hint, or by opening any entry.
- Stored in `localStorage` (`tsumugi.glossaryHintDismissed`); every access
  wrapped in try/catch. Without storage the hint shows again next visit.

## Migration of existing explanations

**ⓘ callout chips** keep their labels and positions; each becomes a
`<Term>` that opens an entry at a section. Their `title` text moves into
the entry sections.

| Card | Chip | Target |
|---|---|---|
| Embeddings | learned, not designed | `embedding` § `learned` |
| Embeddings | no position here | `embedding` § `position` |
| Embeddings | tied with Logits | `embedding` § `tied` |
| Logits | tied with Embeddings | `logit` § `tied` |
| Logits | similar rows, similar scores | `logit` § `similar-rows` |
| Logits | N scores, k shown | `logit` § `top-k` |
| Softmax | why exponentiate | `softmax` § `exp` |
| Softmax | temperature | `softmax` § `temperature` |
| Softmax | top-k only | `softmax` § `top-k` |

`softmaxCallouts` and the `CALLOUTS` arrays in `EmbeddingLookup.tsx` and
`LogitsReadout.tsx` become label → target lists; run-specific wording
moves into the entries' "In this run" blocks.

**Residual-stream diagram:** the attention and MLP boxes lose their
`title` tooltips and become term links to `attention` and `mlp` (summary
on hover or focus, entry on click).

**Card headings:** each stage card's `h3` (Tokenizer, Embeddings,
Transformer layers, Logits, Softmax, Sampler) gets a small ⓘ button that
opens the stage's entry.

**Captions:** the first mention of a glossary concept in each card becomes
a term link; later mentions in the same card stay plain.

**Unchanged:** `title` attributes that label controls (run shelf pin,
pipeline band "Jump to …", grid explorer head labels, compare cycle
chips).

**README:** unchanged except one sentence pointing to the in-app glossary
under the "?" button, plus the screenshot paragraph below.

## Testing

- **Type check:** `GlossaryId` makes unknown term ids a compile error.
- **Registry test:** exactly one entry per id in `GLOSSARY_IDS`, every
  entry has a stage, every summary is one sentence (ends with a full stop
  and contains no other sentence-ending full stop).
- **Section test:** render every entry body, collect `<Section>` ids, and
  check every section used by callout chips and by `<Term section>` in
  entry bodies against them.
- **Entries:** every body renders; every `inRun` renders with a sample
  run summary and returns null without data it needs; a throwing `inRun`
  is skipped without breaking the entry.
- **Store:** open, open another (Back history), Back, close.
- **Term:** summary on hover and on focus; click opens; unknown id renders
  plain text.
- **Drawer:** Escape and × close and return focus; Related and Back
  navigate; section links call `scrollIntoView` (mocked) and highlight;
  index groups by stage in pipeline order.
- **Hint:** shown first time; × persists dismissal; throwing storage
  shows the hint without errors.
- **Existing tests:** Embeddings, Logits and Softmax callout tests switch
  from checking `title` text to checking that the chips open the right
  entry and section.
- **E2E (sim smoke in CI):** open "?", open "How to read Tsumugi", close;
  during a run, open a term from a card and check that "In this run"
  shows the run's numbers.
- **Screenshots:** a `glossary` scene in `npm run screenshots` with the
  drawer open beside the Softmax card, used in a short README paragraph.

## Out of scope

- URL links to entries (`#glossary/<id>`) — planned follow-up.
- The extended entry set (separate head-role entries, BPE, RMSNorm, RoPE,
  quantization, run archive).
- MLP-specific entries (arrive with #15).
- Mobile layout (#9).
- Translations.

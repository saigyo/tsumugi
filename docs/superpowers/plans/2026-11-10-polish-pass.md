# Polish Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 15 defects of the UI review and give every repeated element (buttons, selected states, switches, token chips, bars, floating surfaces, icons, notes, numbers) one definition, guarded by tests.

**Architecture:** All styling stays in `src/index.css`; shared element classes replace the per-component ones, and the global `button` rule shrinks to a reset so no component style can be overridden by it. Three small shared modules are added under `src/ui/` (icons, number formatter, decimal field parsing). A new Playwright "polish check" walks the app in both themes and three engines; its tests are written first, parked with `test.fixme`, and each fixing task switches its test on.

**Tech Stack:** React 19, TypeScript, plain CSS, Vitest + Testing Library, Playwright (Chromium, Firefox, WebKit), oxlint.

**Spec:** `docs/superpowers/specs/2026-11-10-polish-pass-design.md`. Reference sheet with working CSS for every rule: `docs/superpowers/specs/2026-11-10-polish-pass-mockup.html`. The review's own scripts (page-side measuring functions, the walk through simulated and real mode) are kept as reference in `docs/superpowers/plans/2026-11-10-polish-pass-review/`; they are not part of the build.

## Global Constraints

- Branch `polish-pass` in the main checkout. Integrate by rebase and fast-forward only; no merge commits. Do not push.
- Existing `data-testid` values, accessible names, and the state attributes tests read (`aria-pressed`, `aria-current`, `aria-expanded`, `data-active`, `data-selected`, `data-role`, `data-fork`, `data-chosen`, `data-top`) stay. Exception (plan ruling): where a symbol was a button's only name (the playback buttons), its `title` text becomes its `aria-label`.
- Tests change only where a label's wording changes (D8, D10, D11, R9) or a symbol becomes an icon.
- No new dependencies. Icons are inline SVG.
- Every colour comes from a variable defined in both `:root` and `:root[data-theme="dark"]`; the existing guards in `src/styleRules.test.ts` keep passing.
- Not in this plan: column width, gaps between blocks, card paddings, chart widths, animation, the settings line wrapping, run chips shifting sideways on hover.
- No change to what a chart shows or to chart geometry. The only chart colours that change: candidate chips in Logits and Softmax lose their washes, and the row marks in the Logits matrix drawing become plain indigo.
- Corner radii: `2px`, `4px`, `6px`, `8px`, `14px`, `999px` only (`50%` for circles, `0` allowed).
- Commands: `npm test` (unit), `npx tsc -b`, `rtk proxy npx oxlint` (lint; the `npm run lint` wrapper can misreport), `npx playwright test` (browser). `src/App.test.tsx` has two tests that fail rarely under full-suite load; rerun once before treating that as a failure.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (or the committing model's name). Never add the untracked file `:memory:.ses`.

## Review Focus

1. **A very long or unusual token in a chip** (40 characters, `<|im_start|>`, a lone space `␣`, a line break `↵`): the chip keeps one line and one height, and lists with fixed columns show an ellipsis and the full text in a `title`. Test in Task 6.
2. **Odd typing in the Temperature field** (empty, `0.`, `,5`, `2.5`, `abc`, pasted `0,85`): the field never shows a comma, the run parameter never becomes `NaN` or leaves 0 to 2, and an unusable entry keeps the last good value. Tests in Task 9.
3. **Formatter edges** (0, a value that rounds to zero, negative zero, 0.996, 9999 against 10 000, `null`): tests in Task 9.
4. **Icon-only buttons need a name**: every control in the walk has a non-empty accessible name. Polish test `every control has an accessible name`, switched on in Task 3.
5. **Keyboard focus after the button reset**: every control still shows the focus ring in both themes. Polish test `every control shows a focus ring`, switched on in Task 4.

---

### Task 1: The polish check

Review: yes.

**Files:**
- Create: `e2e/polish/probe.ts`, `e2e/polish/walk.ts`, `e2e/polish.spec.ts`, `e2e/fixtureRun.ts`
- Modify: `e2e/steps.spec.ts` (import `openFixtureRun` from `e2e/fixtureRun.ts` instead of defining it), `playwright.config.ts`, `.github/workflows/ci.yml`

**Interfaces:**
- Produces, `e2e/fixtureRun.ts`: `openFixtureRun(page: Page): Promise<void>` (moved unchanged from `e2e/steps.spec.ts`).
- Produces, `e2e/polish/probe.ts`:
  - `installProbe(page: Page): Promise<void>`: an init script that defines `window.__polish`.
  - `type Control = { index: number; key: string; text: string }`; `type ControlState = { background: string; text: string; contrast: number | null; disabled: boolean; hasName: boolean; focusRing: boolean }`
  - in the page: `__polish.controls(): Control[]` (tags one visible representative per distinct control with `data-polish="<index>"`; distinct = tag + classes + state attributes), `__polish.state(index): ControlState`, `__polish.snapshot(): void` and `__polish.movedVertically(exceptIndex): string[]` (other controls, tokens, run chips and stage cards whose top or height changed by more than 0.5px since the snapshot), `__polish.textIssues(): string[]`, `__polish.clippedLabels(): string[]`, `__polish.overflowingTips(): string[]`, `__polish.sidewaysScroll(): number`, `__polish.palette(): Record<string, string>` (resolved values of `--card`, `--washi`, `--white`, `--seg-ground`, `--fill`, `--fill-hover`).
  - Start from `docs/superpowers/plans/2026-11-10-polish-pass-review/lib.mjs`, which has working versions of the colour parsing (`rgb()`, `rgba()`, `color(srgb …)`), effective background, contrast, control tagging and the snapshot comparison.
- Produces, `e2e/polish/walk.ts`:
  - `THEMES = ['light', 'dark'] as const`; `setTheme(page: Page, theme: 'light' | 'dark'): Promise<void>` (emulates the system setting, with no stored choice, and waits 700ms for colour transitions).
  - `walk(page: Page, visit: (state: string) => Promise<void>): Promise<void>`: drives the app and calls `visit` at each of these states, in order: `first-visit`, `sim-run-end`, `sim-embeddings`, `sim-layers`, `sim-logits`, `sim-softmax`, `sim-sampler`, `settings-tip` (pointer on the Temperature info mark), `term-tip` (pointer on the first glossary term in the detail card), `glossary-index`, `glossary-softmax`, `glossary-mlp`, `runs-8` (eight simulated runs, one pinned), `runs-list`, `compare-armed`, `compare-side`, `compare-combined`, `real-layers` (after `openFixtureRun`).
  - The walk pauses playback before each `visit`, so stage cards are not mid-transition.

- [ ] **Step 1: Move `openFixtureRun`** to `e2e/fixtureRun.ts` and import it in `e2e/steps.spec.ts`. Run `npx playwright test e2e/steps.spec.ts`; expected: pass.

- [ ] **Step 2: Write `e2e/polish.spec.ts`** with these tests, each looping over `THEMES` and running the walk once per theme. Names are fixed; later tasks refer to them.

```ts
test('the walk reaches every state', …)                      // collects the labels visit() receives; expects the 18 above, in order
test.fixme('hovered controls stay readable', …)              // each distinct control once per theme: hover it; unless disabled, contrast >= 4.5
test.fixme('hover keeps fills and chart colours', …)         // if the resting background is not card, washi, white, seg-ground or transparent,
                                                             // the hovered background is the same, or fill-hover when the resting one is fill
test.fixme('hovering moves nothing vertically', …)           // snapshot, hover, movedVertically(index) is empty
test.fixme('every control has an accessible name', …)        // state(index).hasName for every control
test.fixme('every control shows a focus ring', …)            // focus each control from the keyboard; outline or box-shadow ring present
test.fixme('numbers follow one format', …)                   // textIssues() is empty in every state
test.fixme('chart labels are not cut', …)                    // clippedLabels() is empty in every state
test.fixme('tooltip text stays inside its box', …)           // overflowingTips() is empty in settings-tip and term-tip
test.fixme('the page never scrolls sideways at 400px and 1380px', …)  // sidewaysScroll() === 0 in every state at both widths
```

`textIssues()` reports visible text matching any of: a hyphen directly before a digit when preceded by start, space, `(`, `~`, `≈`, `+` or `:` (a hyphen used as minus); a digit, a comma and exactly three digits (thousands comma); a digit, a space and `%`; `~` directly before a sign or digit. It skips the prompt field, token text (`.token`, token chips) and `code`.

`clippedLabels()` reports every SVG `text` whose box leaves its `svg` element's box by more than 1px, and every HTML axis label whose `scrollWidth` exceeds its `clientWidth` without an ellipsis.

Set `test.setTimeout(240_000)` for the file.

- [ ] **Step 3: Engines.** In `playwright.config.ts` define three projects: `chromium` (all specs), `firefox` and `webkit` (only `polish.spec.ts` and `formulas.spec.ts`, the latter created in Task 11). In `.github/workflows/ci.yml` change the install line to `npx playwright install --with-deps chromium firefox webkit`.

- [ ] **Step 4: Run** `npx playwright test e2e/polish.spec.ts`. Expected: `the walk reaches every state` passes in all three projects; nine tests skipped per project.

- [ ] **Step 5: Prove the parked tests detect today's defects.** Temporarily change each `test.fixme` to `test`, run the file in the `chromium` project only, and note in the report which fail and with what first message. Expected to fail today: readable, fills, vertically, format, labels, tooltip, sideways. Restore `test.fixme` afterwards. If one of those seven passes today, its detection is wrong: fix it before committing.

- [ ] **Step 6: Commit** `test: polish check walking the app in both themes and three engines`.

---

### Task 2: Corners, the new title size, small controls

Review: no.

**Files:**
- Modify: `src/index.css`, `src/styleRules.test.ts`

**Interfaces:**
- Produces: CSS variable `--t-sub: 1.05rem` in `:root`; class `sub-title` (font size `--t-sub`, weight 600, colour `--ai-deep`).

- [ ] **Step 1: Write the failing guards** in `src/styleRules.test.ts`:

```ts
test('corner radii come from the set', () => {
  const ok = /^(0|2px|4px|6px|8px|14px|999px|50%|2px 2px 0 0)$/
  expect(declared('border-radius').filter((v) => !ok.test(v))).toEqual([])
})
test('bold text is the semibold weight', () => {
  const rule = blocks.find((b) => b.selector === 'strong, b')
  expect(rule?.body).toMatch(/font-weight:\s*600/)
})
```

Extend the two existing type tests: `--t-sub` is allowed in `font sizes come from the scale`, and `the type variables exist with the specified sizes` expects `--t-sub: 1.05rem`.

- [ ] **Step 2: Run** `npm test -- src/styleRules.test.ts`. Expected: the two new tests and the size test fail.

- [ ] **Step 3: Change `src/index.css`.** Map every radius: 1px → 2px; 3px → 4px; 5px → 4px for tokens and small buttons, 8px for boxes that hold text; 6px → 8px except a control nested 2px inside another (switch options, the field inside a settings chip), which stays 6px; 7px → 8px; 10px → 8px; 12px → 999px. Add `strong, b { font-weight: 600; }`, `code { font-family: var(--mono); font-size: var(--t-cap); background: var(--washi); border: 1px solid var(--line); border-radius: 4px; padding: 0 .25rem; }`, `input[type="checkbox"] { accent-color: var(--fill); }`, the `--t-sub` variable and `.sub-title`.

- [ ] **Step 4: Run** `npm test`. Expected: pass.

- [ ] **Step 5: Commit** `feat(style): six corner sizes, a sub-title size, small controls`.

---

### Task 3: Icons

Review: no.

**Files:**
- Create: `src/ui/Icon.tsx`, `src/ui/Icon.test.tsx`
- Modify: every component that renders one of the replaced symbols (list from `grep -rnE "[📌🔗⇩⇧⇄◐☀☾⏺⏸▶◀▾▸ⓘ]" src`), `src/index.css`, `src/styleRules.test.ts`, the tests that assert on a replaced symbol, `e2e/polish.spec.ts`

**Interfaces:**
- Produces: `type IconName = 'pin' | 'pinned' | 'link' | 'download' | 'close' | 'import' | 'compare' | 'theme-auto' | 'theme-light' | 'theme-dark' | 'info' | 'chevron' | 'first' | 'back' | 'play' | 'pause' | 'forward' | 'last' | 'live' | 'prev' | 'next'`; `Icon({ name }: { name: IconName }): JSX.Element` rendering `<svg class="icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">`. Drawings: the `<symbol>` paths in the reference sheet (`i-close`, `i-down`, `i-up`, `i-link`, `i-pin`, `i-pinned`, `i-compare`, `i-auto`, `i-sun`, `i-moon`, `i-info`, `i-chev`, `i-first`, `i-back`, `i-play`, `i-pause`, `i-fwd`, `i-last`, `i-live`, `i-left`, `i-right`). Filled shapes carry `fill="currentColor" stroke="none"` on the path itself.
- Produces: CSS `.icon { width: 1rem; height: 1rem; flex: none; fill: none; stroke: currentColor; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; vertical-align: -0.15em; }`.

- [ ] **Step 1: Write the failing tests.**

`src/ui/Icon.test.tsx`: every `IconName` renders an `svg.icon` with `aria-hidden="true"` and at least one child shape.

`src/styleRules.test.ts`:

```ts
test('no symbol or emoji icons in component code', () => {
  const banned = /[📌🔗⇩⇧⇄◐☀☾⏺⏸▶◀▾▸ⓘ]/u
  // every .ts/.tsx under src except *.test.* and src/test/; reuse the file walk of the existing test
  // 'no colour literals in TypeScript source' (the same walk serves buttonsWithoutClass in Task 4)
  expect(filesWith(banned)).toEqual([])
})
```

- [ ] **Step 2: Run** `npm test -- src/ui src/styleRules.test.ts`. Expected: fail (no module; banned characters found).

- [ ] **Step 3: Implement `Icon`** and replace the symbols. Rules:
  - The "×" of a close or remove button becomes `close`; "×" as a multiplication sign stays text.
  - `ⓘ` becomes `<Icon name="info" />` everywhere: the `Term` icon variant's children, the callout links under the cards (`src/glossary/callouts.ts` and the `embed-callout` buttons), and the glossary text that shows the mark.
  - The run chip's pin uses `pin`, or `pinned` in coral when pinned; the `grayscale`/opacity trick on `.run-chip-pin` goes.
  - The playback buttons get `aria-label` equal to their current `title` (Global Constraints ruling); "Live" keeps its word next to the `live` icon.
  - The steppers in `LayersNav`, `NeuronCard`, `MlpStepsCard` use `prev` / `next` and keep their `aria-label`.
  - Fold labels built in `src/viz/chatFold.ts` drop their arrow; the component that renders them adds `chevron`.
  - "?" on the help button stays a letter.

- [ ] **Step 4: Update tests** that assert on a replaced symbol to assert on the accessible name or test id instead.

- [ ] **Step 5: Switch on** `every control has an accessible name` in `e2e/polish.spec.ts` (remove `.fixme`).

- [ ] **Step 6: Run** `npm test`, `npx tsc -b`, `rtk proxy npx oxlint`, `npx playwright test`. Expected: pass; eight polish tests still skipped.

- [ ] **Step 7: Commit** `feat(ui): drawn icon set replaces text symbols and emoji`.

---

### Task 4: Buttons and the hover rule (R2, D1, D3, D12)

Review: yes.

**Files:**
- Modify: `src/index.css`, `src/styleRules.test.ts`, every component in the mapping below, `e2e/polish.spec.ts`

**Interfaces:**
- Produces: classes `btn` (plain, 2rem high, 8px corners, hairline `--line-strong` border, card ground), `btn-main`, `btn-accent`, `btn-danger`, `btn-quiet`, `btn-lg` (2.5rem), `btn-sm` (1.5rem, 4px corners), `btn-icon` (square). Exact declarations: the `.btn*` rules in the reference sheet (its `primary` is `btn-main`, `accent` is `btn-accent`, and so on).
- Produces: the global `button` rule is only a reset (`font: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer`), and there is no global `button:hover` rule.

Mapping (every `<button>` gets a class; a button not listed keeps its own component class and gets no `btn`):

| Element (test id) | Classes |
| --- | --- |
| `btn-generate` | `btn btn-accent btn-lg` |
| `btn-stop` | `btn btn-danger btn-lg`, rendered before Generate |
| `load-real` (card variant), `banner-load`, "Measure all 30 layers" | `btn btn-main` |
| `btn-compare-arm`, `btn-compare-exit`, `btn-import`, `btn-explore-heads`, "Run with MLP n off and compare", the continue button in `RunEndDetail`, "Copy link" (non-chip variant), `btn-fallback`, `neuron-retry`, `steps-retry`, `embed-geometry-retry`, `btn-live` | `btn` |
| `theme-toggle`, `btn-help`, `gl-close`, the five playback buttons | `btn btn-icon` |
| `toggle-system`, `toggle-examples`, `system-reset` | `btn btn-quiet` |
| the notice close button, `btn-hint-dismiss` | `btn btn-quiet btn-icon btn-sm` |
| `nav-prev`, `nav-next`, `neuron-prev`, `neuron-next`, `steps-prev`, `steps-next` | `btn btn-icon btn-sm` |

The `speed`, `dist-sort` and `grid-sort` drop-downs get class `btn` (same height, border, corners, `color: var(--ink)`).

- [ ] **Step 1: Write the failing guards** in `src/styleRules.test.ts`:

```ts
test('the global button rule is a reset and has no hover rule', () => {
  expect(blocks.filter((b) => /^button(:[\w-]+(\([^)]*\))?)*$/.test(b.selector) && /:hover/.test(b.selector))).toEqual([])
  const base = blocks.find((b) => b.selector === 'button')
  expect(backgroundOf(base!.body)).toEqual(['none'])
})
test('every button in component code names its kind', () => {
  // every "<button" tag in src/**/*.tsx (not tests) has a className attribute
  expect(buttonsWithoutClass()).toEqual([])
})
```

- [ ] **Step 2: Run** the guard file. Expected: both fail.

- [ ] **Step 3: Implement.** Add the `.btn*` rules; reduce the global rule; apply the mapping; delete the per-component button rules they replace (`.btn-generate`, `.btn-stop`, `.load-real`, `.knockout-measure`, `.knockout-run`, `.explore-toggle`, `.hdr-btn`, `.gl-help-btn`, `.gl-close`, `.toggle`, `.sim-banner-close`, `.gl-hint-dismiss`, `.controls button`, `.neuron-stepper button`, `.link-button` where it styled a button kind). Buttons that are not a kind (run chips' parts, token chips, glossary terms and index rows, heatmap and grid cells, knockout bars and chips, candidate rows, switch options) keep their component rules and get an explicit hover rule only where one is wanted; give switch options the class `seg-opt`.
  - D3: in `PromptBar`, Stop renders before Generate; the prompt line keeps Generate's right edge fixed (the prompt field takes the slack).
  - D12: `LoadRealButton`'s card variant renders `btn btn-main`; it is never italic and has one size.
  - The settings-chip fields and the prompt field keep their focus styles.

- [ ] **Step 4: Switch on** `hovered controls stay readable`, `hover keeps fills and chart colours` and `every control shows a focus ring`.

- [ ] **Step 5: Add a unit test** in `src/app/PromptBar.test.tsx`: while generating, `btn-stop` comes before `btn-generate` in the document, and both carry `btn-lg`.

- [ ] **Step 6: Run** `npm test`, `npx tsc -b`, `rtk proxy npx oxlint`, `npx playwright test`. Expected: pass in all three engines; five polish tests still skipped.

- [ ] **Step 7: Commit** `feat(style): four button kinds, no global hover rule`.

---

### Task 5: Selected states, switches, pills (R3, R4)

Review: yes.

**Files:**
- Modify: `src/index.css`, `src/styleRules.test.ts`, `src/app/PromptBar.tsx`, `src/app/RunShelf.tsx`, `src/app/Controls.tsx`, `src/viz/EditSizeCard.tsx`, `src/viz/NeuronCard.tsx`, `src/viz/LayerVotesCard.tsx`, `src/viz/LayersNav.tsx`, `src/viz/AttentionHeatmap.tsx`, `src/viz/EmbeddingLookup.tsx`, `src/viz/LogitsReadout.tsx`, `src/viz/compare/CompareView.tsx`, and their tests

**Interfaces:**
- Produces: state classes `is-view` (background `--fill`, text `--on-ai`; hover `--fill-hover`), `is-picked` (border `--ai`, background `--ai-wash-strong`, text `--ai-deep`, weight 600), `is-chosen` (border `--shu`, background `--shu-wash`). They are added next to the existing state attributes, never instead of them.
- Produces: class `pill` (pill corners, hairline border, card ground, caption size; hover: border `--ai`). One switch class, `seg`, with options `seg-opt`.

- [ ] **Step 1: Write the failing tests.**

Guard in `src/styleRules.test.ts`: no rule other than `.is-view`, `.btn-main` and their hover rules sets `background: var(--fill)` or `var(--fill-hover)`; the selector list `.mlp-seg`, `.cmp-mode-btn`, `.layers-nav-link[aria-current="true"]`, `.cmp-tick[data-selected="true"]` no longer appears in the stylesheet.

Component tests (extend the existing files):
- `PromptBar.test.tsx`: the pressed mode option has `is-view`; with `mode="real"` and the model ready the `mode-real` option's text is "Real model"; while it is loading or not yet loaded it contains "· 250 MB".
- `LayersNav` test: the current link has `is-view`.
- `CompareView` test: the selected `cmp-tick` has `is-view`; the active distribution mode button has `is-view`; the selected `cmp-token` has `is-picked`.
- `AttentionHeatmap` test: the active head chip has `is-picked`.
- `LogitsReadout` test: the picked `readout-candidate` has `is-picked`; the top candidate's chip has `is-chosen` whether or not it is the picked row.
- `RunShelf.test.tsx`: the active chip has `is-view`; with a comparison, the A and B chips have `is-picked` and not `is-view`.

- [ ] **Step 2: Run** those files. Expected: fail.

- [ ] **Step 3: Implement.**
  - Switches: the three `mlp-seg` groups and the compare mode buttons become `seg` with `seg-opt` options (the compare pair is wrapped in one `seg`, `role="group"`, `aria-label="Distribution view"`); the pressed option gets `is-view`.
  - Pills: `layers-nav-link`, `head-chip`, `cmp-head-chip`, `example-chip`, `run-more`, the knockout chips get `pill` alongside their own class. Knockout chips keep their category border colours.
  - Cycle numbers (`cmp-tick`): `btn btn-sm`; selected gets `is-view`; the coral fork border stays.
  - View state: run chip and list row (`data-active`), current token (`token-current` already fills; point it at the same declarations).
  - Picked state: head chips, `edit-row-label`, `vote-row-label`, `neuron-cand`, `embed-token`, the selected `cmp-token`, the picked `readout-candidate`, `btn-live` while following, an open quiet toggle (wash only), the two compare chips and list rows with their A or B letter.
  - Logits: coral marks the top candidate (chip and bar), not the picked row.
  - Hover on a picked or chosen element changes at most its border.

- [ ] **Step 4: Run** `npm test`, `npx tsc -b`, `rtk proxy npx oxlint`, `npx playwright test`. Expected: pass.

- [ ] **Step 5: Commit** `feat(style): three selected looks, one switch, one pill`.

---

### Task 6: Token chips and bars (R5, R6)

Review: yes.

**Files:**
- Create: `src/ui/tokenWash.ts`, `src/ui/tokenWash.test.ts`
- Modify: `src/index.css`, `src/styleRules.test.ts`, `src/viz/details/TokenizerDetail.tsx`, `src/viz/details/SoftmaxDetail.tsx`, `src/viz/details/SamplerDetail.tsx`, `src/viz/EmbeddingLookup.tsx`, `src/viz/EmbeddingGeometry.tsx`, `src/viz/LogitsReadout.tsx`, `src/viz/LayerVotesCard.tsx`, `src/viz/NeuronCard.tsx`, `src/viz/TokenPopover.tsx`, `src/viz/compare/CompareView.tsx`, `src/glossary/entries/token.tsx`, and their tests

**Interfaces:**
- Produces, `src/ui/tokenWash.ts`: `washIndexes(ids: number[]): number[]`. For each position, the wash (0 to 5) of its token: a token id seen before reuses its wash; a new id takes `next % 6`, or the one after it if that equals the wash of the position just before, and `next` continues from the wash taken.
- Produces: class `tok` (pill, monospace text at caption size, hairline border, washi ground, one line), `tok-sm`, `tok-wash-0` … `tok-wash-5` (the six washes, no border), `tok-away` (coral wash), child `tok-id` (label size, soft ink). States from Task 5: `is-picked`, `is-chosen`.
- Produces: class `bar` (10px high, 2px corners, background `--ai`), `bar-chosen` (`--shu`), `bar-neg` (`--shu`). SVG bars that stay SVG use 10px height and `rx="2"` with the same colours.

- [ ] **Step 1: Write the failing tests.**

Guards in `src/styleRules.test.ts`: the selectors `.token-chip`, `.hue-0` … `.hue-5`, `.hue-fill-0` … `.hue-fill-5`, `.mlp-cand`, `.neuron-chip`, `.cmp-bar`, `.popover-bar`, `.votes-bar`, `.neuron-size-bar`, `.bar-rect` are gone; `--vote-pos` and `--vote-pos-deep` are not used for bars (only the glossary vote figure may use them).

Component tests:
- `src/ui/tokenWash.test.ts`: `washIndexes([504, 2644, 2643, 335, 260, 1171, 975, 260, 2644, 436])` equals `[0, 1, 2, 3, 4, 5, 0, 4, 1, 2]` (the repeated 260 and 2644 reuse 4 and 1; 436 would take 1, the wash just before it, so it takes 2); `washIndexes([])` equals `[]`; `washIndexes([7, 7, 7])` equals `[0, 0, 0]`.
- `TokenizerDetail` / `EmbeddingLookup`: prompt-token chips have `tok` and `tok-wash-n` with n from `washIndexes` over the whole prompt (also when the system prompt is folded); two chips with the same token id carry the same wash class.
- `LogitsReadout`, `SoftmaxDetail`, `LayerVotesCard`, `NeuronCard`: candidate chips have `tok` and no `tok-wash-` class.
- `SamplerDetail`: the chosen line contains a `tok is-chosen` chip with the token text and "(id 16394)"-style text after it; no quotation marks.
- `LogitsReadout`, `SoftmaxDetail`: the top candidate's bar has `bar-chosen`; no other bar has it.
- Review Focus 1: a candidate whose text is 40 characters renders with a `title` holding the full text.

- [ ] **Step 2: Run** those files. Expected: fail.

- [ ] **Step 3: Implement.**
  - Chips as in R5 of the spec. The strip (`.token`) is untouched; the two token rows at the top of compare take the strip's look (`token`, `token-generated`) and keep their `cmp-token` test id and state attributes.
  - Compare distribution lists, the token pop-over and the neuron table's word lists use `tok tok-sm`; "away from" uses `tok tok-sm tok-away`.
  - Token row labels of the edit-size chart are axis labels, not chips: they keep their text look and get `is-picked` when selected.
  - Logits matrix row marks: plain `--ai`; the picked candidate's mark is emphasised.
  - Bars as in R6. The chosen or top row's bar gets `bar-chosen`. Keep the existing bar test ids on the element that carries the width.
  - Lists with fixed chip columns keep their maximum width with an ellipsis; the chip gets `title` with the full token text.

- [ ] **Step 4: Run** `npm test`, `npx tsc -b`, `rtk proxy npx oxlint`, `npx playwright test`. Expected: pass.

- [ ] **Step 5: Commit** `feat(style): one token chip, one bar`.

---

### Task 7: Floating surfaces and tooltips (R7, D2, D14)

Review: no.

**Files:**
- Modify: `src/index.css`, `src/styleRules.test.ts`, `src/glossary/Term.tsx`, `src/viz/ResidualStreamDiagram.tsx`, `src/viz/MlpStepsCard.tsx`, `src/viz/TokenPopover.tsx`, `src/app/RunShelf.tsx`, `src/viz/KnockoutCard.tsx`, `e2e/polish.spec.ts`

**Interfaces:**
- Produces: class `pop` (background `--card`, text `--ink`, weight 400, 1px `--line-strong` border, 8px corners, `--shadow-pop`, `white-space: normal`).

- [ ] **Step 1: Write the failing guards:** `--shadow-accent` is used only by `.btn-accent` and the active stage card; `--shadow-tip` is unused and removed from both variable blocks; the rule for `.pop` sets `white-space: normal`.

- [ ] **Step 2: Run** the guard file. Expected: fail.

- [ ] **Step 3: Implement.** `gl-tip`, `step-box`, `token-popover` and `run-list` get `pop` and lose their own surface declarations (they keep position, width and padding). The tooltip's "Click to read more." is soft ink. `knockout-tip` becomes an inline panel: card ground, hairline border, 8px corners, no shadow. The slider thumb uses `--shadow-card`.

- [ ] **Step 4: Switch on** `tooltip text stays inside its box`.

- [ ] **Step 5: Run** `npm test`, `npx playwright test`. Expected: pass.

- [ ] **Step 6: Commit** `feat(style): one floating surface; tooltips wrap`.

---

### Task 8: Titles, notes and messages (R9, D8, D10, D11)

Review: no.

**Files:**
- Modify: `src/index.css`, `src/styleRules.test.ts`, `src/viz/DetailPanel.tsx`, `src/App.tsx`, `src/app/RunShelf.tsx`, `src/viz/compare/CompareView.tsx`, the detail and card components whose titles or notes change, `src/glossary/` entries that quote a changed title, and their tests

**Interfaces:**
- Produces: classes `note` (caption size, `--ink-soft`, upright), `fine` (label size, `--ink-faint`), `readout` (caption size, `--ai-deep` on `--ai-wash`, 8px corners), `error` (caption size, `--shu-deep` on `--shu-wash`, 1px `--shu-line` border, 8px corners, leading `info` icon).
- Produces: `DetailPanel` takes a new prop `generating: boolean`.
- Produces: `data-testid="compare-hint"` in the runs row's notes line.

- [ ] **Step 1: Write the failing tests.**

Guard: `font-style: italic` is declared for no selector except `em` and `i`.

`DetailPanel` tests, exact texts:
- no events, `generating={false}`: "Press Generate, then step through the pipeline."
- no events, `generating={true}`: "Generating… the first token will appear here."
- events present, cursor before the first stage: "Press play or step forward to walk through this run."

`RunShelf.test.tsx`: armed with active run #8 → `compare-hint` reads "Pick a second run to compare with #8."; not armed, or a comparison shown → no `compare-hint`.

`App` or `RunShelf` test: importing a file that is not JSON shows `import-error` with class `error` and text "This file is not a Tsumugi run: not valid JSON."

`CompareView` test: the section titles read "Cycle 2 · Distributions" and "Cycle 2 · Attention".

Detail tests: the Layers card title is "Layers"; the Sampler card title starts with "Sampler (".

- [ ] **Step 2: Run** those files. Expected: fail.

- [ ] **Step 3: Implement.**
  - Titles: sentence case; stage name first; every `h4` title of a card inside the detail card uses `sub-title`.
  - Notes: `mlp-note`, `grid-semantics`, `cmp-badge`, `readout-sum`, `attn-hint`, `tokenizer-note`, `embed-caption` and similar become `note` (or `fine` where they are label size today); notes that are sentences get a capital and a full stop. `votes-readout` and `attn-readout` become `readout`.
  - Errors: `import-error`, `model-error`, `knockout-error`, `neuron-error`, `steps-error`, `embed-geometry-error`, `formula-error` use `error`.
  - Running text in `.detail`, the inner cards and `.compare-view` has `max-width: 72ch`.
  - "Loading vocabulary geometry…" is capitalised in both places.

- [ ] **Step 4: Run** `npm test`, `npx tsc -b`, `rtk proxy npx oxlint`, `npx playwright test`. Expected: pass.

- [ ] **Step 5: Commit** `feat: one style for notes and errors; clearer empty, compare and import messages`.

---

### Task 9: Numbers and the Temperature field (R10, D9)

Review: no.

**Files:**
- Create: `src/ui/format.ts`, `src/ui/format.test.ts`, `src/ui/decimalField.ts`, `src/ui/decimalField.test.ts`
- Modify: every site that formats a number for display (`grep -rnE "toFixed\(|Math\.round\([^)]*\* 100\)|toLocaleString" src`, excluding SVG coordinates), `src/app/PromptBar.tsx`, `src/styleRules.test.ts`, the glossary entries with written-out numbers, `e2e/polish.spec.ts`

**Interfaces:**
- Produces, `src/ui/format.ts`:

```ts
export function count(n: number): string                    // grouping with U+202F from 10 000 up
export function fixed(n: number, digits: number): string    // real minus; a value that rounds to zero has no sign
export function signed(n: number, digits?: number): string  // digits default 2; '+' or '−'; a value that rounds to zero has no sign
export function percent(p: number | null): string           // p in 0..1
export function points(d: number, approx?: boolean): string // d = difference of two probabilities, −1..1
```

- Produces, `src/ui/decimalField.ts`: `normaliseDecimal(text: string): string | null` (comma becomes point; returns `null` unless the result is empty or digits with at most one point) and `decimalValue(text: string, min: number, max: number): number | null` (`null` unless it parses to a finite number within the limits).

- [ ] **Step 1: Write the failing tests.**

```ts
// src/ui/format.test.ts
expect(count(49152)).toBe('49 152'); expect(count(1536)).toBe('1536'); expect(count(9999)).toBe('9999')
expect(count(10000)).toBe('10 000'); expect(count(1234567)).toBe('1 234 567')
expect(fixed(-0.514, 2)).toBe('−0.51'); expect(fixed(-0.001, 2)).toBe('0.00'); expect(fixed(16.36, 2)).toBe('16.36')
expect(signed(4.29)).toBe('+4.29'); expect(signed(-0.15)).toBe('−0.15'); expect(signed(0)).toBe('0.00'); expect(signed(-0.004)).toBe('0.00')
expect(signed(34.94, 1)).toBe('+34.9')
expect(percent(0.94)).toBe('94%'); expect(percent(0)).toBe('0%'); expect(percent(0.003)).toBe('<1%')
expect(percent(1)).toBe('100%'); expect(percent(0.996)).toBe('>99%'); expect(percent(null)).toBe('—')
expect(points(0.63)).toBe('+63 pts'); expect(points(-0.79, true)).toBe('≈ −79 pts'); expect(points(0)).toBe('0 pts')

// src/ui/decimalField.test.ts
expect(normaliseDecimal('0,85')).toBe('0.85'); expect(normaliseDecimal(',5')).toBe('.5'); expect(normaliseDecimal('0.')).toBe('0.')
expect(normaliseDecimal('')).toBe(''); expect(normaliseDecimal('abc')).toBeNull(); expect(normaliseDecimal('1.2.3')).toBeNull()
expect(decimalValue('0.85', 0, 2)).toBe(0.85); expect(decimalValue('.5', 0, 2)).toBe(0.5)
expect(decimalValue('0.', 0, 2)).toBe(0); expect(decimalValue('', 0, 2)).toBeNull(); expect(decimalValue('2.5', 0, 2)).toBeNull()
```

`PromptBar.test.tsx`: typing "0,5" into `temp-input` shows "0.5" and Generate is called with `temperature: 0.5`; typing "2.5" or "abc" leaves the last good value in the parameters; clearing the field keeps the last good value; the field is not `type="number"`.

Guard in `src/styleRules.test.ts`: no file under `src` except `src/ui/format.ts` (tests excluded) defines a local `pct`, `signed` or `percent` helper, and none contains `* 100)}%` or `'~'`.

- [ ] **Step 2: Run** those files. Expected: fail.

- [ ] **Step 3: Implement** the two modules; replace every display site. Counts in prose and labels go through `count` (so "1,536" becomes "1536" and "49 152" keeps its look); the compare delta column uses `points(delta, approx)`; the "N %" label in the step charts loses its space. Titles of chart cells (`<title>`, `title=`) use the formatter too. SVG coordinates keep `toFixed`.

- [ ] **Step 4: The Temperature field** becomes a text field with `inputMode="decimal"`: it holds its own text, accepts a change only when `normaliseDecimal` returns a string, and updates the temperature when `decimalValue(text, 0, 2)` is a number. Limits, test id, label and info mark are unchanged.

- [ ] **Step 5: Switch on** `numbers follow one format`.

- [ ] **Step 6: Run** `npm test`, `npx tsc -b`, `rtk proxy npx oxlint`, `npx playwright test`. Expected: pass.

- [ ] **Step 7: Commit** `feat: one number format; Temperature shows a decimal point everywhere`.

---

### Task 10: Layout defects (D4, D5, D6, D7, D15)

Review: yes.

**Files:**
- Modify: `src/index.css`, `src/viz/AttentionHeatmap.tsx`, `src/viz/EmbeddingGeometry.tsx`, `src/viz/EmbeddingLookup.tsx`, `src/viz/PipelineBand.tsx`, `src/app/RunShelf.tsx`, `src/app/ModelSlot.tsx`, `src/viz/details/SoftmaxDetail.tsx`, `e2e/polish.spec.ts`, `e2e/chat.spec.ts`

- [ ] **Step 1: Switch on** `hovering moves nothing vertically`, `chart labels are not cut` and `the page never scrolls sideways at 400px and 1380px`. Extend the walk with one state, `chat-layers`: a chat-format run on the Layers stage, built with `buildFixtureTrace` from `src/test/fixtures` if it supports a chat format, otherwise a simulated run whose prompt is `<|im_start|>user`. Update `the walk reaches every state` to expect the 19 states. Add to `e2e/chat.spec.ts`: with a chat run at 1024px width the pipeline band's `scrollWidth` does not exceed its `clientWidth`.

- [ ] **Step 2: Run** `npx playwright test e2e/polish.spec.ts e2e/chat.spec.ts --project=chromium`. Expected: the three polish tests and the band test fail.

- [ ] **Step 3: Fix.**
  - D4: the heatmap and the similarity grid size their label margins from the longest label, including special tokens and the folded system-prompt column.
  - D5: the row strip scales to its column; at the desktop column width all cells show without scrolling.
  - D6: at 400px the Softmax list fits (its bar column shrinks); the simulation notice stacks text above its button below 600px; the loop label does not wrap.
  - D7: the band fits the column from 1024px up with the chat example (reduce the gaps or the stage padding, not the text).
  - D15: every run chip state and `run-more` have one fixed height; the action buttons no longer decide it.

- [ ] **Step 4: Run** `npm test`, `npx playwright test`. Expected: pass in all three engines; no polish test is skipped any more.

- [ ] **Step 5: Commit** `fix: clipped chart labels, the cut-off row strip, phone-width overflow, run row height`.

---

### Task 11: Fraction lines in WebKit (D13)

Review: no.

**Files:**
- Create: `e2e/formulas.spec.ts`
- Modify: `src/index.css` or the formula component, as the cause requires

- [ ] **Step 1: Write the failing test** `e2e/formulas.spec.ts` (runs in all three projects): open a simulated run on the Softmax stage; for the first `mfrac` in the card, take a screenshot of its box and assert that some pixel row between the numerator's bottom and the denominator's top is dark across at least 60% of the box width.

- [ ] **Step 2: Run** it. Expected: passes in `chromium` and `firefox`, fails in `webkit`.

- [ ] **Step 3: Find the cause and fix it.** Start with how the bundled Latin Modern Math font reaches the `math` element in WebKit. Take a screenshot of the Softmax formula in `chromium` and `firefox` before and after: they must be pixel-identical. If no fix keeps them identical, stop and report what you found instead of committing a change.

- [ ] **Step 4: Run** `npx playwright test e2e/formulas.spec.ts`. Expected: pass in all three.

- [ ] **Step 5: Commit** `fix: fraction lines in WebKit`.

---

### Task 12: Final walk and README screenshots

Done by the session that runs the plan, not by a subagent: it needs the real model in a headed browser.

- [ ] **Step 1:** Repeat the review's walk with the scripts in `docs/superpowers/plans/2026-11-10-polish-pass-review/` (`sim.mjs`, `real.mjs`, `dark.mjs`, `cross.mjs` for `webkit` and `firefox`), on a copy of `.screenshots-profile`. Every `!!` flag in `report.mjs hover` and `report.mjs text` is explained or fixed.
- [ ] **Step 2:** Look at the screenshots of every state in both themes, and at each of the 15 defects by name.
- [ ] **Step 3:** Compare each element of the reference sheet with the built app, side by side.
- [ ] **Step 4:** Check the README for text that names a changed control or title; run `npm run screenshots`; leave `docs/screenshots/coreference-farmer.png` as it is.
- [ ] **Step 5:** Commit `docs: README screenshots after the polish pass`.

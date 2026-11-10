# Dark Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dark theme that follows the system setting, with an Auto / Light / Dark switch in the header, and charts whose colours run the right way for a dark page.

**Architecture:** The resolved theme is a `data-theme` attribute on `<html>`, set before first paint by an inline script and afterwards by a small store. Dark colours are one `:root[data-theme="dark"]` block overriding the light variables. Chart ramps stop being literal `hsl()` strings in TypeScript and are mixed from each theme's end-point variables.

**Tech Stack:** React 19, TypeScript, plain CSS (`color-mix`), zustand, Vitest + Testing Library, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-10-dark-theme-design.md`
**Trial palette (starting values):** `docs/superpowers/specs/2026-11-10-dark-theme-palette.css`

## Global Constraints

- The light theme looks exactly as it does today. No layout, type or behaviour change.
- Colour literals only inside the two variable blocks of `src/index.css` (`:root` and `:root[data-theme="dark"]`); none in `.ts`/`.tsx` once Task 3 is done.
- Every colour variable in the light block exists in the dark block and the reverse (after Task 4).
- All text is at least 4.5:1 on its ground in both themes.
- Theme choice: `localStorage` key `tsumugi.theme`, values `light` or `dark`; absent means Auto.
- Switch: test id `theme-toggle`; order Auto → Light → Dark → Auto; glyphs ◐ / ☀ / ☾; accessible name and tooltip exactly "Theme: auto (follows system)", "Theme: light", "Theme: dark".
- Existing `data-testid` values and accessible names stay. No new dependencies.
- The existing type rules in `src/styleRules.test.ts` keep passing.
- Each task ends with `npm test`, `npm run lint`, `npm run build` passing; Tasks 2, 3 and 5 also `npm run e2e`.
- Work on branch `feat/dark-theme`; integrate by rebase and fast-forward only. README screenshots are not regenerated.

## Review Focus

1. Storage blocked (private mode, `localStorage` throws): the page still loads, follows the system, and the switch works for the page's life. (Task 2 tests)
2. A stored value that is neither `light` nor `dark`: treated as Auto. (Task 2 test)
3. The system flips between light and dark while the app is open in Auto: the page follows; in Light or Dark it does not. (Task 2 tests)
4. A value outside a ramp's range (attention weight above 1, a negative ratio, a zero peak): the fill is clamped to an end colour, never an invalid colour string. (Task 3 tests)
5. The theme changes while the all-heads thumbnails are on screen: they redraw in the new colours. (Task 3 test, Task 5 look)

---

### Task 1: Split the fill roles (light theme only, no visible change)

**Files:**
- Modify: `src/index.css`, `src/styleRules.test.ts`

**Interfaces:**
- Produces: variables `--fill` (value of today's `--ai-deep`), `--fill-hover` (value of today's `--ai`), `--on-shu` (white). `--on-ai` now means text on `--fill`/`--fill-hover` only.

- [ ] **Step 1: Add failing tests** to the contrast list: `['--on-ai', '--fill']`, `['--on-ai', '--fill-hover']`, `['--on-shu', '--shu']`; and a rule test: no declaration block that sets `color: var(--on-ai)` also sets `background` (or `background-color`) to `var(--ai-deep)`, `var(--ai)` or `var(--shu)`; none that sets `color: var(--on-shu)` uses a background other than `var(--shu)`.
- [ ] **Step 2: Run** `npx vitest run src/styleRules.test.ts`. Expected: new tests FAIL.
- [ ] **Step 3: Reclassify.** Every rule using `--ai-deep` or `--ai` as a background under `--on-ai` text moves to `--fill` / `--fill-hover` (active run chip, current token, selected switch side, `.load-real` and similar buttons, their hover rules). Every rule with text on a `--shu` background uses `--on-shu`. Rules using `--ai-deep` / `--ai` as text, border, stroke or bar colour stay. SVG `fill:` of shapes that carry white text counts as a background.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: pass.
- [ ] **Step 5: Prove no visible change.** Before Step 3 (from `git stash` or the previous commit) and after, screenshot at 1380px a simulated run on each of the six stages plus the open runs list and the glossary; compare the pairs pixel by pixel (any image-diff approach). Expected: identical. Record the result in the report.
- [ ] **Step 6: Commit** `refactor(style): separate fill colours from indigo text, own text colour on vermillion`

### Task 2: Theme mechanism and the switch

**Files:**
- Create: `src/app/theme.ts`, `src/app/theme.test.ts`, `src/app/ThemeToggle.tsx`, `src/app/ThemeToggle.test.tsx`, `e2e/theme.spec.ts`
- Modify: `index.html`, `src/App.tsx`, `src/index.css`

**Interfaces:**
- Produces:

```ts
export type ThemeMode = 'auto' | 'light' | 'dark'
export type Theme = 'light' | 'dark'
export const THEME_KEY = 'tsumugi.theme'
export function readMode(): ThemeMode                       // never throws
export function resolveTheme(mode: ThemeMode, systemDark: boolean): Theme
export function nextMode(mode: ThemeMode): ThemeMode         // auto → light → dark → auto
export const useTheme: UseBoundStore<StoreApi<{ mode: ThemeMode; theme: Theme; cycle(): void }>>
export function initTheme(): () => void                      // applies data-theme, listens to the system; returns a cleanup
export function ThemeToggle(): JSX.Element
```

  `useTheme().theme` is what Task 3's canvas redraw subscribes to.

- [ ] **Step 1: Write failing tests** in `theme.test.ts`:
  - `readMode()` is `'auto'` with nothing stored, `'dark'`/`'light'` for those values, `'auto'` for `'purple'`, and `'auto'` when `localStorage.getItem` throws.
  - `resolveTheme('auto', true) === 'dark'`, `('auto', false) === 'light'`, `('light', true) === 'light'`, `('dark', false) === 'dark'`.
  - `nextMode` cycles `auto → light → dark → auto`.
  - After `initTheme()` with the system dark (mock `matchMedia`): `document.documentElement.dataset.theme === 'dark'`; firing the media query's change to light flips it to `'light'`; after `cycle()` to `'dark'` a system change no longer affects it; `cycle()` stores `light`/`dark` and removes the key on returning to Auto; with `setItem` throwing, `cycle()` still changes `mode` and `data-theme`.
- [ ] **Step 2: Write failing tests** in `ThemeToggle.test.tsx`: the button has test id `theme-toggle`; its accessible name and `title` are "Theme: auto (follows system)" with text ◐; one click → "Theme: light" and ☀; another → "Theme: dark" and ☾; a third → auto.
- [ ] **Step 3: Run** `npx vitest run src/app/theme.test.ts src/app/ThemeToggle.test.tsx`. Expected: FAIL.
- [ ] **Step 4: Implement** `theme.ts` and `ThemeToggle.tsx`; call `initTheme()` once from `App`; render `ThemeToggle` in the header immediately left of the help button (after Compare and Import), styled like them.
- [ ] **Step 5: Inline script in `index.html`**, in `<head>` before any stylesheet or module: read `tsumugi.theme` inside try/catch, fall back to `matchMedia('(prefers-color-scheme: dark)')`, set `document.documentElement.dataset.theme`. In `src/index.css` add `:root[data-theme="dark"] { color-scheme: dark; }` as the (still otherwise empty) dark block, and make the guard test accept colour literals in that block.
- [ ] **Step 6: e2e** `e2e/theme.spec.ts`: with `colorScheme: 'dark'` emulated, `data-theme` is `dark` before the app module loads (check in a `page.addInitScript`-free way: read it at `domcontentloaded`); clicking `theme-toggle` once gives `light`; after reload it is still `light`; two more clicks return to Auto and `dark`.
- [ ] **Step 7: Run** `npm test && npm run lint && npm run build && npm run e2e`. Expected: pass.
- [ ] **Step 8: Commit** `feat(app): theme follows the system, with an Auto / Light / Dark switch`

### Task 3: Chart ramps from theme variables

**Files:**
- Create: `src/viz/ramps.ts`, `src/viz/ramps.test.ts`
- Modify: `src/viz/stripMath.ts`, `src/viz/mlpMath.ts`, `src/viz/AttentionHeatmap.tsx`, `src/viz/AttentionGridExplorer.tsx`, `src/viz/EmbeddingGeometry.tsx`, the callers (`VectorStrip.tsx`, `EmbeddingLookup.tsx`, `MlpStepCharts.tsx`, `EditSizeCard.tsx`, `KnockoutCard.tsx`, `LayerVotesCard.tsx`), their tests, `src/index.css`, `src/styleRules.test.ts`

**Interfaces:**
- Consumes: `useTheme` from Task 2 (canvas redraw only).
- Produces:

```ts
// t in 0..1 (clamped): low → high of the value ramp
export function seqFill(t: number): string   // `color-mix(in hsl, var(--ramp-hi) ${p}%, var(--ramp-lo))`
// v relative to peak (clamped to ±1): each side from its own low end to its full colour
export function divFill(v: number, peak: number): string
// for canvas: the same mixing done in JS from end colours read off the document
export function seqRgb(t: number, lo: string, hi: string): string
```

  `cellFill`, `voteFill` and `ratioFill` keep their names and signatures and delegate to these. Variables: `--ramp-lo`, `--ramp-hi` (exist); `--div-pos-lo`, `--div-pos`, `--div-neg-lo`, `--div-neg` (exist; set `--div-pos` and `--div-neg` to today's full-strength ends `hsl(211 45% 38%)` and `hsl(13 55% 50%)`, low ends `hsl(211 45% 88%)` and `hsl(13 55% 90%)`).

- [ ] **Step 1: Write failing tests** in `ramps.test.ts`:

```ts
expect(seqFill(0)).toBe('color-mix(in hsl, var(--ramp-hi) 0%, var(--ramp-lo))')
expect(seqFill(1)).toBe('color-mix(in hsl, var(--ramp-hi) 100%, var(--ramp-lo))')
expect(seqFill(1.7)).toBe(seqFill(1)); expect(seqFill(-3)).toBe(seqFill(0)); expect(seqFill(NaN)).toBe(seqFill(0))
expect(divFill(5, 10)).toBe('color-mix(in hsl, var(--div-pos) 50%, var(--div-pos-lo))')
expect(divFill(-10, 10)).toBe('color-mix(in hsl, var(--div-neg) 100%, var(--div-neg-lo))')
expect(divFill(3, 0)).toBe(divFill(1, 1))        // zero peak does not divide by zero
expect(seqRgb(0, 'rgb(10, 20, 30)', 'rgb(110, 120, 130)')).toBe('rgb(10, 20, 30)')
expect(seqRgb(1, 'rgb(10, 20, 30)', 'rgb(110, 120, 130)')).toBe('rgb(110, 120, 130)')
```

  And in `src/styleRules.test.ts`: the light `--ramp-lo`, `--ramp-hi`, `--div-*` values are exactly `hsl(211 30% 94%)`, `hsl(211 55% 24%)`, `hsl(211 45% 88%)`, `hsl(211 45% 38%)`, `hsl(13 55% 90%)`, `hsl(13 55% 50%)`; the allow-list of files with colour literals is gone and the scan allows none.
- [ ] **Step 2: Run** the two test files. Expected: FAIL.
- [ ] **Step 3: Implement** `ramps.ts`; make the five files use it. Where a fill was an SVG `fill` attribute, set it through `style={{ fill }}` (a `var()` inside a presentation attribute is not resolved). The canvas in `AttentionGridExplorer.tsx` reads `--ramp-lo`/`--ramp-hi` from `getComputedStyle(document.documentElement)` (resolve them to rgb by assigning to a scratch element's colour if needed), draws with `seqRgb`, and redraws when `useTheme().theme` changes; add a test that the draw effect re-runs on a theme change (spy on `getContext`, as jsdom has no canvas).
- [ ] **Step 4: Update the existing tests** that asserted literal `hsl(...)` fills to the new strings.
- [ ] **Step 5: Run** `npm test && npm run lint && npm run build && npm run e2e`. Expected: pass.
- [ ] **Step 6: Light regression.** Before and after this task, screenshot with the real model (browser profile `.screenshots-profile`, served on the origin `scripts/capture-screenshots.mjs` uses, so the cached model is reused; create no runs if saved ones exist, otherwise remove what you create): the Embeddings stage, the full Layers stage, the Logits stage and the all-heads explorer. Compare pairs: no visible difference (a per-channel difference of 2 or less from rounding is acceptable; state the largest difference found).
- [ ] **Step 7: Commit** `refactor(viz): chart ramps mixed from theme variables`

### Task 4: The dark palette

**Files:**
- Modify: `src/index.css`, `src/styleRules.test.ts`

**Interfaces:**
- Consumes: the variables from Tasks 1 and 3, the dark block from Task 2.

- [ ] **Step 1: Add failing tests:** the set of colour variable names in the dark block equals the set in the light block (colour variables = those whose light value contains a colour literal or a shadow); every contrast pair in the list is checked against both blocks.
- [ ] **Step 2: Run** `npx vitest run src/styleRules.test.ts`. Expected: FAIL.
- [ ] **Step 3: Fill the dark block** from the trial palette file, with these changes from the spec: `--ai-deep` light blue (text); `--ai` mid-light blue; `--fill` about `#3566c9`; `--fill-hover` a step lighter that still gives white 4.5:1; `--on-ai` white; `--on-shu` near-black; value ramp low end close to the card colour and high end bright light blue; each diverging side from close to the card colour to bright blue / bright coral; six muted dark dye tones in the same hue order; amber mark stays amber. Adjust any value until every contrast pair passes.
- [ ] **Step 4: Run** `npm test && npm run lint && npm run build`. Expected: pass.
- [ ] **Step 5: Commit** `feat(style): dark palette`

### Task 5: Look at everything in dark, fix, document

**Files:**
- Modify: `src/index.css` and any component needing a class for a dark fix; `README.md`

- [ ] **Step 1: Dark screenshots** at 1380px with the theme forced dark, real model: main view with a chat run; each of the six stages (Layers full page, with a measured knockout sweep and a neuron selected); the all-heads explorer; compare view with two runs; run-end detail; glossary drawer with two entries that contain figures and formulas; token popover; runs list; open system prompt and examples rows; simulation notice; model-loading and error notices if reachable; 400px width.
- [ ] **Step 2: Go through them one by one** and fix: unreadable or low-contrast text, glaring light areas, charts where low values do not recede, borders that vanish, shadows that look wrong, the six glossary figures, focus rings. Anything hard-wired to look right only on a light ground gets a variable. Record in the report a table: screen → looked at → problems → fixed or left (with reason).
- [ ] **Step 3: Re-check light:** the same list in the light theme shows no change from before this task.
- [ ] **Step 4: Switch live** between the three modes on the Layers stage with the all-heads explorer open: everything recolours, thumbnails included (Review Focus 5).
- [ ] **Step 5: README:** one short paragraph on the theme switch.
- [ ] **Step 6: Run** `npm test && npm run lint && npm run build && npm run e2e`. Expected: pass.
- [ ] **Step 7: Commit** `feat(style): dark theme fixes from a full visual pass` (README separately if convenient)

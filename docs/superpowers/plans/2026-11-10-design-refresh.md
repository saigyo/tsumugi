# Design Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rearrange the top of the app into header, prompt box and one runs row, and put the whole app on one type system and one set of colour variables, matching the approved mockup.

**Architecture:** Styling lives in the single stylesheet `src/index.css`; two guard tests read that file and keep colours and type on variables. The prompt bar and run shelf are restructured in place, with one new pure function deciding which runs the row shows. Compare and Import move from the run shelf to the app header.

**Tech Stack:** React 19, TypeScript, plain CSS, Vitest + Testing Library (jsdom), Playwright e2e.

**Spec:** `docs/superpowers/specs/2026-11-10-design-refresh-design.md`
**Mockup (the visual reference):** `docs/superpowers/specs/2026-11-10-design-refresh-mockup.html`

## Global Constraints

- Existing `data-testid` values and accessible names stay. Tests change only where structure really moves.
- No behaviour change to generating, saved runs, compare, share links, glossary, playback.
- No change to what any chart shows, to chart geometry, or to colour meanings inside charts.
- No new dependencies, no web fonts, no theme switch, no dark theme.
- After Task 1: no hex, `rgb(`, `rgba(` or `hsl(` outside the one `:root` block of `src/index.css`, and none in `.ts`/`.tsx` source.
- After Task 2: every `font-size` in `src/index.css` is one of `var(--t-title)`, `var(--t-body)`, `var(--t-cap)`, `var(--t-label)`, `var(--t-tick)`, `var(--t-brand)`, `var(--t-prompt)` or `inherit`; every `font-family` is `var(--sans)`, `var(--mono)`, `var(--mark)` or `var(--math)`; every `font-weight` is 400, 600 or (brand only) 800.
- Monospace only for literal model content: token text, token ids, tensor shapes such as `[38×576]`, and variable names that mirror the formulas (`x_last`, `E[1041]`). Everything else is sans; numbers in columns or changing in place get `font-variant-numeric: tabular-nums`.
- Text contrast is at least 4.5:1 on its ground.
- Each task ends with `npm test`, `npm run lint` and `npm run build` passing. Tasks 3–5 and 8 also run `npm run e2e`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Work on branch `feat/design-refresh`; integrate by rebase and fast-forward only.

## Review Focus

1. A run older than the five newest is active and gets removed: the row falls back to the five newest and the "+N more" count is right. (Task 5, `shelfRow` test)
2. Import fails while no runs are saved: the error is visible although the runs row is not. (Task 3 test)
3. A share link with a custom system prompt: the system prompt row is open and shows the text. (Task 4 test)
4. Switching to Simulated with the system prompt row open, then back to Real model: the row and its text come back. (Task 4 test)
5. The full runs list is open and Escape is pressed: it closes and focus returns to the "+N more" button; removing runs until none are hidden also closes it. (Task 5 tests)

---

### Task 1: Colours on variables

**Files:**
- Modify: `src/index.css`
- Create: `src/styleRules.test.ts`

**Interfaces:**
- Produces: one `:root` block (merge the second one near line 558 into it) holding every colour, including new `--white`, `--ai-mute` (soft indigo for text and ⓘ marks; `--ai-soft` stays for fills), `--shadow-card`, `--shadow-box`, `--shadow-accent`, `--shadow-pop`, `--grain` (the body texture colour), `--shu-line` (banner border). `src/styleRules.test.ts` exports nothing; Task 2 adds tests to it.

- [ ] **Step 1: Write the failing tests** in `src/styleRules.test.ts`. Read the stylesheet with `readFileSync('src/index.css', 'utf8')`; strip comments; split off the `:root { … }` block.

```ts
test('there is exactly one :root block', () => { expect(css.match(/:root\s*\{/g)).toHaveLength(1) })
test('no colour literals outside :root', () => {
  expect(rest.match(/#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/g) ?? []).toEqual([])
})
test.each([
  ['--ink', '--washi'], ['--ink-soft', '--washi'], ['--ink-soft', '--card'],
  ['--ink-faint', '--washi'], ['--ink-faint', '--card'],
  ['--ai-mute', '--white'], ['--ai-mute', '--washi'], ['--ai-mute', '--card'],
  ['--on-ai', '--ai-deep'], ['--on-ai', '--shu'], ['--ai-deep', '--ai-wash'], ['--note-ink', '--card'],
])('%s on %s reaches 4.5:1', (fg, bg) => { expect(contrast(vars[fg], vars[bg])).toBeGreaterThanOrEqual(4.5) })
```

`vars` is parsed from the `:root` block; `contrast` is the WCAG relative-luminance ratio for two 6-digit hex values, written in the test file.

- [ ] **Step 2: Run** `npx vitest run src/styleRules.test.ts`. Expected: FAIL (two `:root` blocks, literals outside, `--ink-faint` and `--ai-mute` short or missing).
- [ ] **Step 3: Move every literal into `:root`** and reference it by variable. Darken `--ink-faint` until it passes on both grounds; define `--ai-mute` to pass; keep hues. If white on `--shu` falls short, darken `--shu` by the smallest step that passes. Replace text and ⓘ uses of `--ai-soft` with `--ai-mute`; leave fills and dots on `--ai-soft`.
- [ ] **Step 4: Check source files:** `grep -rnE "#[0-9a-fA-F]{6}\b|rgba?\(" src --include='*.tsx' --include='*.ts' | grep -v test` prints nothing; fix any hit by moving the colour to a CSS class.
- [ ] **Step 5: Run** `npm test && npm run lint && npm run build`. Expected: all pass.
- [ ] **Step 6: Commit** `refactor(style): every colour through a variable, contrast checked`

### Task 2: One type system

**Files:**
- Modify: `src/index.css`, `src/styleRules.test.ts`
- Modify as needed: components under `src/viz/`, `src/app/`, `src/glossary/` where a class has to change to move text between sans and mono

**Interfaces:**
- Produces: in `:root`: `--sans` (today's app font stack, taken from `.app`), `--mono` (unchanged), `--mark` (the 紬 serif stack), `--math` (the formula stack), `--t-title: 1.25rem`, `--t-body: .95rem`, `--t-cap: .82rem`, `--t-label: .72rem`, `--t-tick: .66rem`, `--t-brand` (today's brand size), `--t-prompt: 1.3rem`. Utility class `.num { font-variant-numeric: tabular-nums; }`.

- [ ] **Step 1: Add failing tests** to `src/styleRules.test.ts`:

```ts
test('font sizes come from the scale', () => {
  const bad = [...css.matchAll(/font-size:\s*([^;}]+)/g)].map((m) => m[1].trim())
    .filter((v) => !/^(var\(--t-(title|body|cap|label|tick|brand|prompt)\)|inherit)$/.test(v))
  expect(bad).toEqual([])
})
test('font families come from the four variables', () => { /* same shape: var(--sans|--mono|--mark|--math) */ })
test('font weights are 400, 600 or 800', () => { /* same shape; also rejects `bold` and any `font:` shorthand other than `font: inherit` */ })
```

The declarations inside `:root` and inside `@font-face` are excluded from these three checks.

- [ ] **Step 2: Run** `npx vitest run src/styleRules.test.ts`. Expected: the three new tests FAIL.
- [ ] **Step 3: Map every size** to the nearest role using the spec's table: card headings (`.detail h3`, `.mlp-card h4`) → title; explanatory text, bar labels, token chips → body; captions, notes → cap; small labels and stage subtitles → label; SVG axis text, shape labels, pills → tick. `em`-relative sizes become the role that matches their rendered size.
- [ ] **Step 4: Sort every `font-family: var(--mono)` use** by the monospace rule in Global Constraints. Numbers, percentages, axis labels and captions move to sans with `tabular-nums`. Token text, ids, shapes and formula-mirroring names stay mono.
- [ ] **Step 5: Run** `npm test && npm run lint && npm run build`. Expected: all pass.
- [ ] **Step 6: Look at it.** Start `npm run dev`, generate a simulated run, open each of the six stages and the glossary; fix anything that overflows or collides at the new sizes.
- [ ] **Step 7: Commit** `refactor(style): five text sizes, two fonts with fixed jobs`

### Task 3: Header with Compare and Import

**Files:**
- Create: `src/app/HeaderTools.tsx`, `src/app/HeaderTools.test.tsx`
- Modify: `src/App.tsx`, `src/app/RunShelf.tsx`, `src/app/RunShelf.test.tsx`, `src/index.css`, affected `src/App*.test.tsx` and `e2e/*.spec.ts`

**Interfaces:**
- Produces:

```ts
export function HeaderTools(p: {
  runCount: number; canArm: boolean; selecting: boolean
  onArmCompare(): void; onExitCompare(): void; onImportFile(file: File): void
}): JSX.Element
```

  It renders `btn-compare-arm` (only when `runCount > 1` and not `selecting`; disabled when `!canArm`), `btn-compare-exit` (when `selecting`, text "× Exit compare"), `btn-import` with the hidden `import-input` (always). Titles as today.
- `RunShelfProps` loses `onArmCompare`, `onExitCompare`, `onImportFile`. `RunShelf` no longer returns `null` for zero records when it has a note to show: with no records it renders only the notes line (`import-error`, `shelf-note`).

- [ ] **Step 1: Write failing tests.** `HeaderTools.test.tsx`: import is present with `runCount: 0`; compare absent with one run, present with two, disabled when `canArm` is false; exit replaces arm when `selecting`; choosing a file calls `onImportFile`. `RunShelf.test.tsx`: `records: []` with `importError: 'bad file'` shows `import-error`; the shelf no longer contains `btn-import` or `btn-compare-arm`.
- [ ] **Step 2: Run** `npx vitest run src/app/HeaderTools.test.tsx src/app/RunShelf.test.tsx`. Expected: FAIL.
- [ ] **Step 3: Implement** `HeaderTools` and move the buttons out of `RunShelf`.
- [ ] **Step 4: Header layout in `App.tsx` and CSS**, as in the mockup: left the mark, name and `ModeBadge` as a pale indigo pill, tagline on its own line; right the "New here?" hint, then `HeaderTools`, then `HelpButton`, as quiet outlined buttons.
- [ ] **Step 5: Run** `npm test && npm run lint && npm run build && npm run e2e`; update tests that reached Compare or Import through the shelf. Expected: all pass.
- [ ] **Step 6: Commit** `feat(app): compare and import in the header`

### Task 4: Prompt box

**Files:**
- Modify: `src/app/PromptBar.tsx`, `src/app/PromptBar.test.tsx`, `src/App.tsx`, `src/index.css`, affected `src/App*.test.tsx`, `e2e/*.spec.ts`

**Interfaces:**
- Consumes: nothing new.
- Produces: `PromptBarProps` gains `hasRuns: boolean` (`App` passes `records.length > 0`). New test ids: `toggle-system`, `toggle-examples` (buttons with `aria-expanded`), `system-row`, `examples-row`.

Rules to implement:
- Examples row open = the person's last toggle if they toggled, else `!hasRuns && !ranOnce`, where `ranOnce` becomes true on the first `onGenerate` call from this bar.
- System row: toggle and row exist only in real mode with Chat on. Initially open only when `initial.format` is chat with a system prompt that differs from `DEFAULT_SYSTEM_PROMPT`. The open state and the text survive a switch to Simulated and back.
- Toggle text: "System prompt ▾", or "System prompt · edited ▾" when the text differs from the default; "Examples ▾".
- Chip label "Max tokens" (input keeps `data-testid="maxtok-input"` and its glossary link); other labels unchanged. Inputs, limits and value handling unchanged.

- [ ] **Step 1: Write failing tests** in `PromptBar.test.tsx`:
  - `hasRuns: false` → `examples-row` shown; after Generate → hidden; `hasRuns: true` → hidden, `toggle-examples` shows it.
  - Toggling examples open then generating leaves it open.
  - Real + Chat: `system-row` hidden by default; `toggle-system` shows `system-input`; editing it makes the toggle read "System prompt · edited".
  - `initial` with a custom chat system prompt → `system-row` open with that text. (Review Focus 3)
  - Open the system row, type "Be brief", rerender with `mode: 'sim'` → no `toggle-system`; rerender with `mode: 'real'` → `system-input` shown with "Be brief". (Review Focus 4)
  - The Max tokens field is labelled "Max tokens".
- [ ] **Step 2: Run** `npx vitest run src/app/PromptBar.test.tsx`. Expected: new tests FAIL.
- [ ] **Step 3: Restructure the markup:** one `.prompt-box` holding (1) prompt field, Generate, Stop; (2) the settings line: mode switch, format switch, the four setting chips, then the two toggles at the right end; (3) `system-row` and `examples-row` when open. Existing tests that type into `system-input` or click `example-chip` open the row first where needed.
- [ ] **Step 4: Style it** after the mockup: white box, 1.5px `--ai-deep` border, `--shadow-box`; Generate filled `--shu` with `--shadow-accent`; setting inputs look like plain text until hovered or focused (mockup state "Editing a setting in place"); the Seed input fits ten digits. Remove the old `.prompt-row`, `.system-row`, `.config-row`, `.example-chip-row` rules that no longer apply.
- [ ] **Step 5: Run** `npm test && npm run lint && npm run build && npm run e2e`. Expected: all pass.
- [ ] **Step 6: Commit** `feat(app): prompt box with settings line, system prompt and examples on demand`

### Task 5: Runs row

**Files:**
- Create: `src/app/shelfRow.ts`, `src/app/shelfRow.test.ts`
- Modify: `src/app/RunShelf.tsx`, `src/app/RunShelf.test.tsx`, `src/index.css`, affected `src/App*.test.tsx`, `e2e/*.spec.ts`

**Interfaces:**
- Produces:

```ts
// newest first; the five newest plus any of `keep` that are older, each in its place
export function shelfRow<T extends { id: string }>(
  oldestFirst: T[], keep: Array<string | null | undefined>, limit?: number, // limit defaults to 5
): { row: T[]; hidden: number }
```

- `RunShelf` props otherwise as left by Task 3. New test ids: `run-more` (button, text `+N more ▾`, `aria-expanded`), `run-list` (the full list), `run-list-row`, `run-list-main`.

- [ ] **Step 1: Write failing tests** in `shelfRow.test.ts`, with records `r1…r8` oldest first:

```ts
expect(shelfRow(r, []).row.map(id)).toEqual(['r8','r7','r6','r5','r4'])
expect(shelfRow(r, []).hidden).toBe(3)
expect(shelfRow(r, ['r2']).row.map(id)).toEqual(['r8','r7','r6','r5','r4','r2'])
expect(shelfRow(r, ['r2']).hidden).toBe(2)
expect(shelfRow(r, ['r2','r1','r7']).row.map(id)).toEqual(['r8','r7','r6','r5','r4','r2','r1'])
expect(shelfRow(r.slice(0, 3), ['gone', null]).row.map(id)).toEqual(['r3','r2','r1'])   // Review Focus 1: unknown ids ignored
expect(shelfRow(r.slice(0, 3), []).hidden).toBe(0)
```

- [ ] **Step 2: Write failing tests** in `RunShelf.test.tsx`:
  - Eight records: five `run-chip`s, newest first; `run-more` reads "+3 more ▾"; five or fewer records → no `run-more`.
  - Active run `#2` of eight → six chips, "+2 more ▾"; then rerender without `#2` and `activeId: null` → five chips, "+3 more ▾". (Review Focus 1)
  - Clicking `run-more` shows `run-list` with eight `run-list-row`s; clicking a row's `run-list-main` calls `onActivate` with its id and closes the list.
  - Escape closes the list and focus is on `run-more`; rerendering with five records while the list is open removes the list. (Review Focus 5)
  - While `armed`, clicking a list row calls `onSelectCompareB` and closes the list.
  - Chip text for a chat run at T=0.8 is `#16 · What is the`; its `title` still contains the temperature; a knockout run's chip ends `· MLP 1 off`. The list row text is `#16 · Chat · What is the · T=0.8`.
  - Each list row has pin, export and remove buttons that call the same handlers.
- [ ] **Step 3: Run** `npx vitest run src/app/shelfRow.test.ts src/app/RunShelf.test.tsx`. Expected: FAIL.
- [ ] **Step 4: Implement** `shelfRow`, then the shelf: a "Runs" label, the row from `shelfRow(records, [activeId, compare?.aId, compare?.bId])`, `run-more`, the list (popover or inline panel; closes on Escape and outside click), and the notes line underneath (`share-copied`, `share-link-text`, `import-error`, `shelf-note`).
- [ ] **Step 5: Style it** after the mockup states "Saved runs" and "+3 more": one row, `flex-wrap: nowrap`, `overflow-x: auto`, `run-more` outside the scrolling part so it stays visible; active chip filled `--ai-deep`; A/B compare marks restyled for the new chips. The four action buttons take no space on a resting chip and show when the chip is active, hovered, or has focus inside it (`:focus-within`; the main button stays focusable so Tab reaches them). A pinned resting chip shows a small pin mark.
- [ ] **Step 6: Run** `npm test && npm run lint && npm run build && npm run e2e`; update tests that assumed oldest-first order (for example `run-chip-main').first()` in `e2e/archive.spec.ts`). Expected: all pass.
- [ ] **Step 7: Commit** `feat(app): saved runs in one row with a full list`

### Task 6: Token strip, pipeline, playback and notices

**Files:**
- Modify: `src/index.css`; `src/viz/TokenStream.tsx`, `src/viz/PipelineBand.tsx`, `src/app/Controls.tsx`, `src/app/ModelSlot.tsx` only if a class is missing for the styling

**Interfaces:** none.

- [ ] **Step 1: Restyle** after the mockup: current token filled `--ai-deep` with `--on-ai` text; active stage filled `--shu`, `--on-ai` text, `--shadow-accent`, its subtitle readable on the fill; other stages plain cards; shape labels and the "+1 token" loop unchanged in content. Playback buttons and slider in the quieter rounded style. `ModelSlot`'s three states and the `link-note` after the mockup's "Simulation notice", between the prompt box and the runs row.
- [ ] **Step 2: Run** `npm test && npm run lint && npm run build`. Expected: all pass (the guard tests from Tasks 1 and 2 cover the new CSS).
- [ ] **Step 3: Look at it** in `npm run dev`: step through a run and confirm the active stage and current token are readable at every stage, in the chat token strip too.
- [ ] **Step 4: Commit** `style: active stage and current token filled, quieter playback controls`

### Task 7: Detail cards

**Files:**
- Create: `src/viz/StagePill.tsx`
- Modify: `src/viz/PipelineBand.tsx` (export the stage list), `src/viz/DetailPanel.tsx`, `src/viz/DetailPanel.test.tsx`, `src/index.css`; `src/viz/details/LogitsDetail.tsx`, `SoftmaxDetail.tsx`, `SamplerDetail.tsx` (or the components they use) only if a class is missing for the top-row styling

**Interfaces:**
- Produces: `export const STAGES` from `PipelineBand.tsx` (the existing array); `StagePill({ stage }: { stage: Exclude<StageId, null> })` rendering `<span data-testid="stage-pill" className="stage-pill">Stage {n} of {STAGES.length}</span>`.

- [ ] **Step 1: Write failing tests** in `DetailPanel.test.tsx`: at a logits event the panel shows exactly one `stage-pill` reading "Stage 4 of 6"; at a layer event exactly one, "Stage 3 of 6"; none in the empty state; none at `run-end`.
- [ ] **Step 2: Run** `npx vitest run src/viz/DetailPanel.test.tsx`. Expected: FAIL.
- [ ] **Step 3: Implement** the pill in `DetailPanel` so it sits at the top right of the first card of the active stage (one wrapper around the stage's detail is enough; the stage components need not change).
- [ ] **Step 4: Card surfaces** after the mockup: `--card` ground, 14px radius, hairline border, `--shadow-card`, for `.detail` and the cards inside the Layers stage, the compare view's sections, the run-end detail and the glossary drawer's sections. Layout inside the cards does not change.
- [ ] **Step 5: Top or chosen row** in the Logits, Softmax and Sampler cards: where today the row is outlined in vermillion, also fill its bar with `--shu` on a `--shu-wash` row. No other chart colour changes.
- [ ] **Step 6: Run** `npm test && npm run lint && npm run build`. Expected: all pass.
- [ ] **Step 7: Commit** `feat(viz): stage marker and card surfaces`

### Task 8: Compare with the mockup, screenshots, README

**Files:**
- Modify: `scripts/capture-screenshots.mjs`, `README.md`, `docs/screenshots/*`, `src/index.css` for fixes found

**Interfaces:** none.

- [ ] **Step 1: Screenshot the built app** at 1380px wide in the mockup's state: eight saved runs, a chat run, cursor on Logits. Also capture: a hovered chip, the open runs list, a focused setting, system prompt and examples rows open, the simulation notice. Screenshot the mockup file the same way.
- [ ] **Step 2: Compare section by section** (header, prompt box, runs row, token strip, pipeline, playback, card, each state) and fix every difference in spacing, weight, colour or alignment that the spec does not explain. Charts fill the available width. Record what was compared and what was fixed in the task report.
- [ ] **Step 3: Check a narrow window** (700px and 400px): nothing overlaps, the runs row scrolls sideways, the settings line wraps.
- [ ] **Step 4: Update `scripts/capture-screenshots.mjs`** for the moved controls, run `npm run screenshots`, and update the README text that describes the top area.
- [ ] **Step 5: Run** `npm test && npm run lint && npm run build && npm run e2e`. Expected: all pass.
- [ ] **Step 6: Commit** `docs: screenshots and README for the refreshed design`

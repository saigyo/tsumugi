# Design refresh: tidy top area, one type system

Approved mockup: `docs/superpowers/specs/2026-11-10-design-refresh-mockup.html`
(open it in a browser). It is the reference for the look. Where this text and
the mockup disagree, this text wins. One known flaw in the mockup: the runs
row is cut off at the right edge; the "+N more" button must stay visible.

## Intent

The user finds the top of the app messy and crowded once several runs are
saved and a chat prompt has run: seven stacked rows of controls, all at the
same weight, three of them saved runs. The detail cards are already good, but
they mix fonts in a way that looks random. Overall the app looks "slightly
boring" and lacks "punch".

Success: with eight saved runs and a chat run on screen, everything above the
token strip fits in three blocks (header, prompt box, one runs row); the
prompt and Generate are clearly the main thing; every card follows the same
type rules; and the built page matches the mockup when the two are compared
side by side in screenshots.

## Goals

- Rearrange the top area as in the mockup.
- One type system across the whole app.
- Stronger, more deliberate use of the vermillion accent.
- Detail cards adopt the new surfaces, spacing and type without changing what
  they show.
- Leave the styles ready for a later dark theme.

## Non-goals

- No dark theme now. No theme switch.
- No change to what any chart shows, to chart geometry, or to colour meanings
  inside charts (for example indigo for positive and vermillion for negative
  bars stay).
- No change to behaviour: generating, saved runs, compare, share links,
  glossary and playback work as today.
- No new dependencies and no web fonts. The sans font stays the one the app
  uses now (my call).
- No redesign for phones beyond keeping today's narrow-screen behaviour
  working.

## Anti-goals

- Controls that are harder to reach than today. Everything hidden must be one
  click away and reachable by keyboard.
- A look that drifts from the paper-and-indigo identity. Direction A was
  chosen because it keeps it.

## The top area

### Header

- Left: the 紬 mark, "Tsumugi", the mode badge as a pale indigo pill, and the
  tagline on its own line below.
- Right: Compare, Import, and the help button, as quiet outlined buttons.
  The "New here?" hint sits to the left of these three buttons (my call).
- Compare and Import move here from the run shelf. Import is always shown,
  also with no saved runs (today it needs at least one). Compare is shown when
  there are two or more runs; while a comparison is being picked or shown it
  reads "× Exit compare", as today (my call).

### Prompt box

One white box with an indigo border holds:

1. The prompt field and the Generate button. Generate is filled vermillion.
   While generating it shows the activity label as today, and Stop sits
   beside it.
2. A settings line: the Simulated / Real model switch, the Chat / Continue
   text switch (real mode only), then Temperature, Top-k and Max tokens (both
   modes, as today) and Seed (real mode only, as today) as compact chips, each "label value ⓘ". The label "Max new
   tokens" is shortened to "Max tokens" on the chip; the glossary keeps its
   wording. Each value is still the same input element as today (same limits,
   same test ids, same handling of typed values: no new commit or cancel step),
   restyled to look like plain text until focused or hovered. The Seed chip
   is wide enough for a ten-digit value.
3. At the right end of the settings line, two toggles: "System prompt ▾" (chat
   only) and "Examples ▾". Each opens a row inside the box, below the settings
   line. Both rows can be open at once.

My calls on the toggles:

- The examples row starts open when the shelf is empty at page load (restored
  runs count, so a returning visitor with saved runs sees it closed). It
  closes when the first run starts. Once the person uses the toggle, their
  choice holds for the rest of the page's life.
- The system prompt row is closed by default. When the system prompt differs
  from the default, the toggle reads "System prompt · edited", and a share
  link that carries a custom system prompt opens the row.
- The "Reset to default" link stays in the system prompt row.

The model loading progress, the simulation notice, the load error and the
link-mismatch note keep their content and sit between the prompt box and the
runs row, restyled as in the mockup's "Simulation notice".

### Saved runs

One row, never wrapping, labelled "Runs".

- Order: newest first (today: oldest first).
- The row shows the five newest runs. The active run, and the two runs of a
  comparison, are shown as well when they are older, each in its place in
  the order. A "+N more ▾" button follows when some runs are not in the row;
  N is the number of runs not in the row. It opens a list of all runs (the
  mockup's popover shows fewer; all runs is right). Choosing a run from the
  list activates it and closes the list. An old pinned run is reached
  through the list like any other. Count-based, not measured, so it
  behaves the same at every width (my call). On a narrow window the row
  scrolls sideways.
- A resting chip shows the real/simulated dot, the number, the first words of
  the prompt, and "MLP n off" for a knockout run. Temperature and "Chat" leave
  the chip; they stay in the tooltip and appear in the full list.
- The active chip is filled indigo.
- Pin, copy link, download and remove show only on the active chip and on a
  chip that is hovered or has keyboard focus inside it. A pinned run always
  shows a small pin mark. On touch screens, where nothing hovers, the active
  chip's buttons are the way in (my call).
- Rows in the full list show the full label and the same four buttons.
- Compare selection works on chips and on list rows alike; picking the second
  run from the list closes it and shows the comparison. The A and B marks
  keep today's styling idea, adapted to the new chips.
- "Link copied", the fallback link field, the import error and the "archive
  not persisted" note appear on a line under the runs row.

### Token strip, pipeline, playback

Same content and behaviour. Changes:

- The current token is filled indigo.
- The active pipeline stage is filled vermillion with white text and a soft
  shadow. Other stages are plain cards. The shape labels between stages and
  the "+1 token" return arrow stay.
- Playback buttons and the slider get the quieter rounded style in the mockup.

## Type rules (whole app)

Two fonts with fixed jobs:

- **Sans** for everything a person reads: headings, body, labels, captions,
  notes, chart axis text, percentages and other numbers. Numbers that sit in
  columns or change in place use equal-width digits
  (`font-variant-numeric: tabular-nums`).
- **Monospace** only for literal model content: token text, token ids, and
  tensor shapes such as `[38×576]`, and variable names that mirror the
  formulas (`x_last`, `E[1041]`).
- The serif 紬 mark and the maths font in formulas stay as they are.

Five sizes, defined once as variables, replacing the roughly 25 sizes in use:

| Role | Size | Used for |
| --- | --- | --- |
| Card title | 1.25rem, semibold, deep indigo | the heading of each detail card |
| Body | 0.95rem | explanatory text, bar labels, token chips |
| Caption | 0.82rem, soft ink | captions, notes, settings line, run chips |
| Label | 0.72rem | small labels, stage subtitles; uppercase with letter spacing when it names a chart part |
| Tick | 0.66rem | chart axis ticks, shape labels, pills |

Every existing size maps to the nearest role. SVG chart text (now 9.5–12px)
uses Tick or Label. Larger one-off sizes stay only for the brand name and the
prompt field. Two weights: regular and semibold; bold only for the brand name.

## Colour and surfaces

- The palette stays: washi ground, indigo, vermillion, the six dye washes.
- Vermillion means "this is the action" or "this is where you are": Generate,
  the active stage, and the top or chosen row in the Logits, Softmax and
  Sampler cards, whose bar is filled vermillion where today the row is only
  outlined. It is not added anywhere else, and existing chart meanings for it
  are untouched.
- Detail cards: card surface, 14px radius, hairline border, a soft shadow,
  and a "Stage n of 6" pill (my addition, approved in the mockup). The pill
  appears once per stage view, at the top right of the first card the detail
  panel shows for the active stage; the further cards of the Layers stage do
  not repeat it. The number comes from the same stage list the pipeline band
  uses. No pill on the run-end detail or when no stage is active.
- Dark-theme readiness: every colour in the stylesheet and in component code
  goes through a CSS variable. No hex or rgb values outside the `:root`
  block, including shadows and SVG fills set from TypeScript. The handful of
  literal values in the mockup (shadows, the banner border) and the page's
  paper-texture gradients become variables.

## Constraints

- Existing `data-testid` values and accessible names stay, so the tests keep
  their meaning. Tests change only where structure really moves (run order,
  the "+N more" list, the toggles, Compare and Import in the header).
- New behaviour gets tests: the five-plus-active rule, the list, the two
  toggles and their defaults, chip buttons reachable by keyboard.
- Contrast: all text, including the faint labels and the ⓘ marks, meets
  4.5:1 on its ground. Where the mockup's faint ink or soft indigo falls
  short, darken that variable; the palette's hues stay.
- The compare view, the glossary drawer and the run-end detail follow the same
  type rules and surfaces; their layout does not change.

## Left to the builder

- Exact spacing, shadow strengths and hover styles, guided by the mockup.
- Whether the full-runs list is a popover or an inline panel, as long as it
  closes on Escape and on an outside click.
- How to split the stylesheet work into steps.
- Updating `scripts/capture-screenshots.mjs` for moved controls, regenerating
  the README screenshots, and adjusting README text that describes the top
  area.
- Before calling it done: screenshot the built app in the mockup's state and
  compare it with the mockup, section by section.

# Dark theme

Trial palette: `docs/superpowers/specs/2026-11-10-dark-theme-palette.css`.
It was injected into the live app as a throwaway and the user judged the
result "quite good". It is the starting point for the dark values, not a
finished palette: this document changes parts of it (filled items, chart
ramps), and the contrast checks may force further small changes.

## Intent

The app has one light theme. The user wants a dark theme that follows the
computer's light/dark setting, with a switch in the app to override it. The
colours follow the "night indigo" direction from the redesign mockups: a dark
indigo ground, lighter cards, a coral accent.

Success: with the system in dark mode the app opens dark with no white flash;
every screen and chart is readable; charts read the right way round for a dark
page; and the light theme looks exactly as it does today.

## Goals

- A dark theme covering the whole app: top area, all stage cards, compare
  view, run-end detail, glossary drawer and its figures, popovers, notices.
- Follow the system setting by default and live; a three-way override.
- Charts inverted for dark: low values recede into the card, high values are
  bright.
- The automated colour checks cover both themes.

## Non-goals

- No change to the light theme's look. No layout, type or behaviour change.
- No theme in share links or run files.
- README screenshots stay light.
- No third theme, no per-chart colour options.

## Anti-goals

- A white flash before the dark page appears.
- A chart that is merely dimmed and reads backwards (pale cells for low
  values on a dark card).
- Any text below 4.5:1 contrast in either theme.

## Behaviour

### Choosing the theme

- Three modes: **Auto** (follow the system, the default), **Light**, **Dark**.
- In Auto the app follows `prefers-color-scheme` and changes live when the
  system changes.
- The choice is stored in this browser (`localStorage`, key `tsumugi.theme`,
  values `light` or `dark`; absent means Auto). If storage is unavailable the
  switch still works for the page's life.
- The resolved theme is set as `data-theme="light"` or `"dark"` on the `<html>`
  element by a small inline script in `index.html` that runs before the page
  first draws. `color-scheme` follows, so form controls and scrollbars match.

### The switch (my call on the details)

- An icon button in the header, immediately left of the help button, in the
  same quiet outlined style. Test id `theme-toggle`.
- Each click moves to the next mode: Auto → Light → Dark → Auto.
- It shows one icon per mode (◐ auto, ☀ light, ☾ dark; plain text glyphs, no
  new dependency). Its accessible name and tooltip state the mode: "Theme:
  auto (follows system)", "Theme: light", "Theme: dark".

## Colours

### Structure

- The light values stay in `:root`. The dark values live in one block,
  `:root[data-theme="dark"]`. Every colour variable defined in the light block
  has a dark value; fonts and sizes are not repeated.
- Dark starting values come from the trial palette file.

### Filled items (decided with the user)

In the trial, one indigo variable served as both text colour and fill, which
turned the active run and current token into heavy light-blue pills. So the
roles are split:

- A fill colour for indigo-filled items (the active run chip, the current
  token, the selected side of a switch, indigo buttons such as "Load the real
  model"), separate from indigo used as text (card titles, links). Light:
  both are today's deep indigo, so nothing changes. Dark: the fill is a medium
  blue (about `#3566c9`) with white text; indigo text is light blue.
- Text on coral/vermillion fills (Generate, the active stage) gets its own
  variable, separate from text on indigo fills. Light: white, as today. Dark:
  a near-black, because white on coral is too hard to read.
- The coral accent keeps its soft glow in dark (the trial's shadow values).

The roles after the split (my call):

| Role | Light | Dark |
| --- | --- | --- |
| Indigo as text and strong lines (`--ai-deep` today) | unchanged | light blue |
| Indigo for borders, bars and links (`--ai` today) | unchanged | mid-light blue |
| Indigo fill (new) | today's `--ai-deep` value | medium blue, about `#3566c9` |
| Indigo fill, hovered (new) | today's `--ai` value | a step lighter, still 4.5:1 with white |
| Text on indigo fills (`--on-ai`) | white | white |
| Text on coral fills (new) | white | near-black |

Every rule that today uses `--ai-deep` or `--ai` as a background under
`--on-ai` text moves to the fill variables; rules using them as text, border
or bar colour stay.

Trial values for variables this document does not mention (washes, lines,
soft tones) stand unless a contrast check fails. The contrast pair list is
the existing one, run once per theme, plus: white on the indigo fill and on
its hover, and the coral-fill text on coral.

### Charts (decided with the user)

In dark mode light colours read as high and dark colours as low, the opposite
of the light theme. So:

- **Value ramps** (attention heatmap, the all-heads thumbnails, the "how much
  each layer changes" grid, the embedding similarity grid): low is close to
  the card colour, high is bright light blue.
- **Positive/negative ramps** (the votes grid, the vector strips in the
  Embeddings, Logits and MLP cards): zero is close to the card colour,
  positive runs to bright blue, negative to bright coral.
- Legends (the small gradient bars) use the same end points, so they always
  match their chart.
- The six token washes become six muted dark tones with the same hue order.
- Markers, notes, the amber neuron mark and the vote lines get dark values
  that keep their meaning (amber stays amber, the selected line stays coral).

How (constraint from the code): these ramps are today computed in TypeScript
as literal `hsl()` strings in five files (`src/viz/stripMath.ts`,
`src/viz/mlpMath.ts`, `src/viz/AttentionHeatmap.tsx`,
`src/viz/AttentionGridExplorer.tsx`, `src/viz/EmbeddingGeometry.tsx`). They
must instead be built from the theme's end-point variables (the existing
`--ramp-lo/-hi` and `--div-*`), so they change with the theme. In the light
theme the result must be the same colour as today for every value: the
current ramps interpolate hue-constant HSL linearly, so mixing the two end
points in HSL reproduces them. The positive and negative sides keep their own low
end points, as today and as in the trial palette. "Same as today" means no
visible difference in light-theme screenshots plus a unit test that the two
ends of each ramp resolve to exactly today's end colours; whole-percent
rounding in between need not be reproduced. The canvas thumbnails cannot use
CSS mixing;
they read the end points from the computed style and redraw when the theme
changes. After this, the allow-list of files with colour literals in the
style guard test is empty and removed.

## Checks

- The style guard tests (`src/styleRules.test.ts`) are extended:
  - colour literals are allowed only in the light and dark variable blocks;
  - every colour variable in the light block exists in the dark block and
    the reverse;
  - every contrast pair is checked in both themes, including the new fill and
    on-fill pairs;
  - no colour literals in `.ts`/`.tsx` at all.
- Unit tests for the theme logic: stored choice, Auto following the system,
  live change, the cycle order, storage failure.
- A browser test: with the system emulated dark, the page's background is
  dark at first paint and the switch overrides it; the choice survives a
  reload.
- A light-theme regression check: screenshots of the main view and each stage
  card before and after the ramp change show no visible difference.
- Before calling it done: dark screenshots of every stage (with the real
  model, so the Layers cards have data), the compare view, the glossary and
  the popovers, looked at one by one for unreadable or glaring parts.

## Left to the builder

- Exact dark values, within the directions above and the contrast checks.
- Names of the new variables.
- Where the theme logic lives and how the header button gets it.
- Whether the six glossary figures need individual dark fixes.

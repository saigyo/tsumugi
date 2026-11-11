# Polish pass: fix what is broken, one rule per element

Reference sheet: `docs/superpowers/specs/2026-11-10-polish-pass-mockup.html`
(open it in a browser; it has a light/dark button). For each element it shows
today's styles on the left and the agreed rule on the right, with working
CSS. Where this text and the sheet disagree, this text wins.

## Intent

A full review of the app (every stage in simulated and real mode, compare,
glossary, runs, hover and focus on every control, both themes, four widths,
three browser engines) found that the app is coherent at first glance but
does not hold up to a careful look: some things are plainly broken, and the
same kind of element is drawn in several different ways.

The user's goal, in their words: "when a user walks through the app nothing
should feel off or out of place, everything should fit perfectly and
harmoniously together. The user should never have the impression that
something is amiss or looks or feels sloppy."

Success: every defect in Part 1 is gone in light and dark, in Chromium,
Firefox and WebKit; every element in Part 2 has one look wherever it
appears; and automated checks keep it that way.

## Goals

- Fix the defects in Part 1.
- Give each repeated element one definition (Part 2) and use it everywhere.
- Add checks that fail when a new inconsistency of these kinds is introduced.

## Non-goals (a later round)

- Page column width, the gaps between blocks, card paddings.
- Making charts fill their cards; the tall token list in Embeddings for chat
  runs; blank space above the compare heatmaps; the pipeline band's width.
- The settings line wrapping in real mode; run chips being wider than their
  label; neighbours shifting sideways when a run chip is hovered.
- Animation, including stage cards and glossary sections lagging on a theme
  switch.
- Checking on Windows.

## Anti-goals

- A change to what any chart shows or to chart geometry. Colour meanings
  inside charts stay, with the two exceptions named in R5.
- A control that is harder to reach, or a changed test id or accessible name.
- Any visible change to formulas in Chromium or Firefox while fixing D13.
- New dependencies. Icons are inline drawings in the code.

## Part 1: defects

Each item says what is wrong today and what "fixed" means.

**D1. Hover changes or destroys a control's look.** One global rule gives
every button a pale indigo background and an indigo border on hover, and it
overrides the component styles. Seen:

- the selected cycle number and the active "side by side / combined" button
  in compare become white on pale (unreadable in light; in dark they lose
  their fill) and stay that way right after a click, while the pointer rests
  on them;
- "Measure all 30 layers" becomes unreadable in light and loses its fill in
  dark;
- knockout bars turn the colour of the card and vanish under the pointer;
- knockout chips, the selected candidate row in Logits, candidate buttons in
  the votes and neuron cards, and the coloured token chips in Embeddings lose
  their colour and their selected outline;
- cells in the all-heads grid get an indigo border where a coral one was
  intended; Stop gets the indigo wash where a coral one was intended.

Fixed: the hover rules in R2 to R5 hold for every control in both themes.
Hover never makes text fall below 4.5:1 and never removes a fill, a
selection mark or a chart colour.

**D2. Tooltip text spills out of its box** on the info marks in the settings
line: the last sentence hangs outside the box. Fixed: every tooltip's text
wraps inside its box.

**D3. Stop lands under the pointer.** When Generate turns into
"Generating…", it widens and Stop appears where the pointer is, so a double
click stops the run. Stop is also 29px tall beside the 39px Generate. Fixed:
Stop sits to the left of Generate and has its height; after clicking the
middle of Generate the pointer is still over the (disabled) Generate button.

**D4. Chart labels are clipped.** Row labels in the attention heatmap (Layers
card and compare) and in the Embeddings similarity grid lose their first
character for `<|im_start|>` and `<|im_end|>`; the column label of the folded
system prompt is cut at the top. Fixed: with the chat example ("What is the
capital of France?") no chart label is cut by the edge of its drawing.

**D5. The embedding row strip is cut off.** In the Embeddings card the strip
under "row n" shows about 40% of its cells; the rest scrolls sideways inside
a narrow column. Fixed: at the desktop column width all cells are visible
without scrolling.

**D6. Phone width (400px).** Fixed means:

- no stage makes the whole page scroll sideways (today: Softmax by 64px,
  Layers with a real run by 7px);
- the simulation notice stacks its text above its button (today the text is
  squeezed into a column about 130px wide);
- "+1 token" on the loop arrow stays on one line and clear of the line.

**D7. The pipeline band is wider than the column** by 8px in a chat run at
desktop width and scrolls sideways. Fixed: with the chat example the band
fits the column at 1024px and wider.

**D8. Stale text in the empty detail card.** It always reads "Press Generate,
then step through the pipeline." Fixed, three cases (wording is my call):

- no run loaded: the text above, unchanged;
- a run is being computed: "Generating… the first token will appear here.";
- a run is loaded and playback is at its start: "Press play or step forward
  to walk through this run."

**D9. Decimal comma.** On a German system the Temperature field shows "0,8"
while the rest of the app shows "T=0.8". Fixed: the field shows a decimal
point on every system. It keeps its limits (0 to 2), its test id, and takes
effect as you type, as today. A typed comma is accepted and shown as a point
(my call). The Up and Down arrow keys still change the value by 0.1 within the
limits, as in the number field it replaces and in Top-k and Max tokens.

**D10. Compare gives no visible instruction.** After clicking Compare the
only hint is a tooltip. Fixed: a line under the runs row reads "Pick a second
run to compare with #n." until a run is picked or compare is left.

**D11. The import error does not look like an error.** It is small grey text
such as "not valid JSON". Fixed: it uses the error style of R9 and a full
sentence: "This file is not a Tsumugi run: " followed by the existing reason.

**D12. "Load the real model" differs by place.** In the Layers cards it is
italic and larger than in the simulation notice. Fixed: one button, R2.

**D13. Fraction lines are missing in WebKit** in the Softmax card formula and
in the Softmax and MLP glossary entries. Seen in Playwright's WebKit build;
not yet confirmed in Safari itself. Fixed: the fraction line is drawn in
WebKit. If that cannot be done without changing how formulas look in
Chromium or Firefox, stop and report instead.

**D14. The coral glow sits on neutral things in dark:** the slider thumb and
the neuron tooltip. Fixed: the glow appears only on Generate and the active
stage; the thumb gets a neutral shadow and the tooltip the surface of R7.

**D15. Hovering a run chip moves the page.** The row grows 1 to 3px taller
and everything below shifts. Fixed: all run chips, in every state, and the
"+N more" button have one height, and hovering a chip moves nothing below
the runs row.

## Part 2: one rule per element

The sheet shows each rule. The mapping lists say where each applies; a
control not listed takes the kind that fits its job.

### R1. Corners

Six values and no others: 2px (bars, heatmap cells), 4px (strip tokens,
small buttons, small badges), 6px (a control sitting 2px inside another, such
as a switch option or the field inside a settings chip), 8px (buttons,
fields, drop-downs, pop-ups, notices, stage cards), 14px (cards, the prompt
box), and the full pill (chips, pills, run chips). Generate and the stage
cards go from 10px to 8px.

### R2. Buttons

Four kinds, three sizes.

| Kind | Look | Used for |
| --- | --- | --- |
| Accent | coral fill, soft shadow | Generate, nothing else |
| Main | deep indigo fill, white semibold text | "Load the real model" (notice and cards), "Measure all 30 layers" |
| Plain | card ground, hairline border | Compare / Exit compare, Import, theme, help, the playback buttons, Live, "Explore all heads", "Run with MLP n off and compare", "Copy link", retry buttons, the glossary close button, Stop (coral text) |
| Quiet | no border, indigo text | "Examples", "System prompt", "Reset to default", the notice and hint close buttons |

Sizes: regular 32px high; large 40px (Generate and Stop); small 24px (the
previous/next steppers in the Layers bar and in card titles, which are one
size everywhere).

Hover: plain and quiet get the pale indigo wash (plain also an indigo
border); main goes one step lighter; accent darkens slightly; Stop gets a
coral wash and border. Text links (underlined, such as "Load the real model"
inside a sentence, "← Back", "All entries") stay links.

The drop-downs (speed, sort) take the plain button's height, border, corner
and text colour.

### R3. Selected

Three looks with three meanings.

| Look | Meaning | Used for |
| --- | --- | --- |
| Deep indigo fill, white text | which view or place you are in | the option chosen in every switch; the current "Jump to" pill; the selected cycle number in compare; the active run chip and list row; the current token in the strip |
| Indigo border, pale indigo wash, deep indigo semibold text | the item you picked to inspect | head chips; token row labels in the edit-size chart; the candidate picker in the votes and neuron cards; the picked token chip in Embeddings; the picked candidate row in Logits; the selected token in compare; Live while following; an open "Examples" / "System prompt" toggle (wash only) |
| Coral | what the model chose, and where the action is | Generate; the active stage; the chip and bar of the chosen or top candidate; the chosen slice in the Sampler |

Consequences (my calls):

- A coral outline no longer means "picked". In Logits the top candidate
  keeps its coral bar and chip mark, and the row picked for the bench gets
  the picked look; they may be the same row.
- The two runs of a comparison get the picked look plus their A or B letter,
  not a coral outline.
- A picked prompt-token chip keeps its wash and gets the indigo border and
  semibold text.
- Marks that belong to a chart keep their colours and their own selected
  marking: knockout bars and chips (coral, blue, grey by category), the
  coral fork marks in compare.
- "· 250 MB" is shown on the "Real model" option only until the model is
  loaded.

### R4. Switches

One design, the prompt box's: a recessed track with the chosen option
filled. Used for Simulated / Real model, Chat / Continue text, MLP /
attention (and "both"), "behind the vote" / "firing most", and "side by
side" / "combined".

### R5. Tokens (decided with the user: option A)

- The token strip keeps its look: it is the text being written. The two
  token rows at the top of compare use the same look.
- Everywhere else a token is one chip: a pill with the token text in the
  monospace font at caption size. A small variant is used in dense lists
  (the neuron table's word lists, the compare distribution lists, the token
  pop-over).
- Colour: the six washes appear only on prompt-token chips (Tokenizer,
  Embeddings, the glossary's token example). They are given out by word, in
  order of first appearance (decided with the user): each new token takes the
  next wash, skipping the wash of the chip just before it, and a repeated
  token reuses the wash it already has. So the two "cat" chips of a prompt
  match, as they are the same token and look up the same row. Candidate chips
  carry no colour, in every card and at every step.
  The user chose this over a colour per candidate because a colour per
  candidate could hold only within one step.
- States on a chip follow R3: picked, or chosen (coral border and wash).
- The Sampler reads "chose [chip] (id n)", not quoted text.
- The "pushes away" word chips in the neuron table keep their coral wash.

Two chart colours change as a result: candidate chips in Logits and Softmax
lose their rotating washes, and the row marks in the Logits matrix drawing,
which were tinted to match, become plain indigo with the picked candidate's
mark emphasised.

### R6. Bars

One bar for probabilities, scores and sizes: 10px high, 2px corners, the
working indigo; coral for the chosen or top row. Used in the Logits, Softmax
and neighbours lists, the token pop-over, both compare lists, the votes
breakdown table, and the neuron table. Positive and negative bars in the
vote tables keep indigo and coral in this shape; the indigo is the same one.
The paired bars of the edit-size chart, the step charts and the knockout
bars are charts and stay.

### R7. Floating surfaces

One surface for everything that floats: the card colour, a hairline border,
8px corners, the pop-up shadow, normal ink. Used for the glossary tooltip
(today a dark box that turns into a light box in the dark theme), the neuron
tooltip in the step charts, the token pop-over and the runs list. The
knockout detail line is not floating: it becomes a plain inline panel with
8px corners. The browser's own tooltips stay on chart cells and as button
hints.

### R8. Icons

A drawn set, one weight, one size (16px; 13px inside small buttons), in the
text colour. It replaces the text symbols and both colour emoji: pin, pinned,
copy link, download, close, import, compare, the three theme states, info,
chevron, and the playback set (first, back, play, pause, forward, last,
live). The sheet has a drawing for each. The info mark "ⓘ" becomes the drawn
info icon everywhere, including the callout links under the cards. Letters
and mathematical signs stay text: "?", "→", "×" as a multiplication sign,
"Σ", "⌀", and the token renderings "␣" and "↵".

### R9. Titles, notes and messages

- Titles start with a capital; nothing else is forced to capitals. Compare's
  section titles become "Cycle n · Distributions" and "Cycle n · Attention".
- A stage card's title starts with the stage's name: "Layers", "Sampler
  (top-k)", "Sampler (greedy)".
- A card inside a card gets a smaller title (1.05rem, semibold, deep
  indigo). This adds one step to the type scale.
- Notes are upright soft-grey text at caption size; the finest print is one
  step smaller and fainter. No italic notes and no indigo notes. Italic stays
  for emphasis inside a sentence. Notes that are sentences start with a
  capital and end with a full stop.
- A read-out line that changes on hover (votes, attention) is one style: the
  pale indigo bar.
- An error is one style: coral text on a coral wash with a hairline border
  and the info icon.
- Running text in cards stops at about 72 characters per line, in every card.

### R10. Numbers

One formatter, used by every card, chart label, tooltip and glossary entry.

| What | Rule | Example |
| --- | --- | --- |
| Large counts | a thin space from 10 000 up; four digits stay together; no commas | 49 152 · 1536 |
| Negative numbers | a real minus sign | −0.51 |
| Approximate values | "≈" before the number | ≈ −79 pts |
| Differences between percentages | points, not percent | +63 pts |
| Percentages | no space; "<1%" for a non-zero value that rounds to zero; "0%" only for an exact zero | 94% · <1% |
| Decimals | a point on every system | 0.8 |
| Pushes and changes | always signed; totals unsigned | +4.29 · 16.36 |

### R11. Small controls

Tick boxes use the indigo fill colour. Bold text uses the app's semibold
weight. Inline code uses the app's monospace font at slightly reduced size
on a faint ground with 4px corners.

## Checks

- The style guard tests are extended: corner radii come only from R1; bold
  is the semibold weight; the new title size is part of the type scale; the
  replaced symbols and emoji do not appear in component code.
- The formatter has unit tests for every row of R10.
- A new browser check walks the app: first visit, a simulated run through
  every stage, the glossary index and two entries, eight runs with the list
  open, compare armed and shown in both modes, and an imported real run on
  the Layers stage (the existing test fixture). In light and in dark it
  asserts:
  - every distinct control, hovered: text contrast at least 4.5:1 (disabled
    controls excepted), and nothing else on the page moves vertically;
  - a filled or chart-coloured control keeps its fill when hovered;
  - visible text has no hyphen used as a minus sign, no comma as a thousands
    separator, no space before "%";
  - no chart label is cut by its drawing; tooltip text stays inside its box;
  - at 400px and at 1380px the page does not scroll sideways.
- This check runs in Chromium, Firefox and WebKit.
- Before calling it done: repeat the review's walk with the real model, in
  both themes and all three engines, look at the screenshots, and compare
  each element with the sheet side by side. Regenerate the README
  screenshots; the coreference specimen stays as it is.

## Constraints

- Test ids and accessible names stay. Tests change only where a label's
  wording changes (D8, D10, D11, R9) or a symbol becomes an icon.
- Every colour still comes from a variable defined for both themes, and the
  contrast guard still passes in both.
- The Temperature field keeps its behaviour apart from D9.

## Left to the builder

- Names and file layout for the shared styles, the icon set and the
  formatter; whether elements are classes or small components.
- The order of work, and how to migrate without a half-converted state on
  `main`.
- Exact icon drawings, starting from the sheet's.
- How the guard and the browser check detect each condition.
- Whether Firefox and WebKit run in CI or as a local command, if they prove
  unreliable on the CI machine. Default: in CI.
- The fix for D13.

## As built: where the result differs from the text above

Decisions taken while building, each with its reason.

- **D13 did not reproduce.** WebKit draws the fraction lines on the current
  build. It is guarded by a test; nothing in the app changed for it.
- **D7 depends on the system's fonts.** On macOS the band fits a chat run by
  one pixel; with the wider fonts of Linux it was 59px too wide, which the
  first CI run showed. The cards now give up side padding (down to 45% of
  it) until the band fits its column, and only then does the band scroll.
  The loop drawing no longer keeps the band wide after one wide step. The
  test also runs with a wider font.
- **D9.** A temperature typed beyond 0 to 2 takes effect as the limit, and on
  leaving the field it shows the value in effect. Up and Down step by 0.1.
- **D11.** A file that imports with damaged parts left out shows a note, not
  an error.
- **R3.** "Live while following" means: playback is playing and the run is
  still being written.
- **R5.** On a washed chip the token id is in the normal text colour (the
  soft grey is 4.1:1 on the washes). A prompt longer than six distinct words
  can put two different words with the same wash side by side when a word
  repeats right after the colours wrap around.
- **R8.** The pinned pin and all other coral text use the deep coral
  (`--shu-deep`); plain coral is 4.35:1 on the card. Coral stays for fills,
  borders and arrows.
- **R9.** The attention read-out shows a hint while nothing is hovered, and
  read-outs are as wide as their text. Switch options and the compare table's
  labels are capitalised like every other label.
- **Colours.** In the light theme the faintest grey (`--ink-faint`) now has
  the value of the soft grey; it was 4.3:1 on the coral and indigo washes.
  Two variables nothing used any more were removed.
- **Chart labels** longer than 22 characters are cut with an ellipsis; the
  read-out and the cell titles keep the full text.
- **Focus.** The per-control focus-ring check runs in Chromium only, because
  script focus does not show the keyboard ring in Firefox and WebKit; a
  second check tabs through the page in all three engines.
- **Heights.** The button sizes make the settings line 42px (was 36), the
  prompt box 117px (was 111) and the header buttons 32px (was 30).
- **R3, R5.** The chosen token is marked on its chip (`is-chosen`) in every
  list that shows it: the Logits and Softmax candidates, the token pop-over
  and both compare lists. Its bar is the only other mark.
- **R10.** One number formatter: `thousands()` is gone and `count()` in
  `src/ui/format.ts` is the only place that groups digits (a narrow no-break
  space from 10 000 up, so 1536 stays together and 49152 is "49 152"). A test
  fails when another file groups digits.
- **D9.** In the Temperature field the arrow keys step to the next multiple
  of 0.1 (from 1.25 up to 1.3, down to 1.2), as a number field does. The
  field also reads full-width digits, point and comma, and ignores spaces
  around the number.
- **Polish check.** It runs with transitions switched off (reduced motion),
  so no wait has to outlast a transition. It reaches the Knockouts card with
  bars of all three chip categories and a flipping bar, the chat run's
  Embeddings card, and the borders of hovered controls.
- **Where the checks run.** CI runs the Chromium browser tests without the
  walk (`npm run e2e`), so a push stays short; the deploy waits for CI only.
  The walk in Chromium (`npm run e2e:design`) runs beside CI on every push,
  in the "Design check" workflow; a failure there shows after the deploy.
  Firefox and WebKit run on request: `npm run e2e:browsers` locally, or the
  "Other browsers" workflow started by hand, after larger changes to the
  design.

# Chat templating — design

**Date:** 2026-10-21
**Status:** Approved design, pre-implementation
**Backlog:** #8 (Chat templating)
**Mockups (real tokens, answers and attention weights):**
- prompt bar, token strip, example chip: `.superpowers/brainstorm/59075-1792614513/content/chat-mode.html`
  (option B in sections 1 and 2; the strip's fold is amended below)
- attention matrix: `.superpowers/brainstorm/59075-1792614513/content/attention-matrix.html` (option B)
- the real model made prominent: `.superpowers/brainstorm/59075-1792614513/content/real-mode-hint.html` (option A)

## Purpose

Tsumugi runs SmolLM2-135M-**Instruct**, but feeds it the prompt as bare text
(`worker.ts`: `tokenizer.encode(prompt, { add_special_tokens: false })`). The
model then continues a document instead of answering a request. Every chat
assistant wraps the user's message in the model's chat template before the
model sees it. This work adds that as a second way to run the real model, and
shows the template's tokens instead of hiding them.

## Findings

From the fp32 model (throwaway scripts). Greedy decoding unless noted;
"sampled" means the app's defaults (temperature 0.8, top-k 10), 200 runs:

- **The template** (from the tokenizer's `chat_template`): each message is
  `<|im_start|>{role}\n{content}<|im_end|>\n`; the generation prompt adds
  `<|im_start|>assistant\n`. When the first message is not a system message,
  the template inserts `You are a helpful AI assistant named SmolLM, trained
  by Hugging Face` as the system turn.
- **Token counts:** "What is the capital of France?" becomes 37 tokens with
  the default system prompt, 7 of them the user's words. The system turn,
  from `system` to the `\n` after its `<|im_end|>`, is the system prompt's own
  tokens plus 4 (20 for the default's 16). `<|im_start|>` and `<|im_end|>`
  are single special tokens; `assistant` splits into `ass` + `istant`.
- **Answers, greedy:** in Chat mode the model answers "The capital of
  France is Paris." and then writes `<|im_end|>`. As plain text, the same
  question gets `<|im_end|>` as its first token and the run ends with nothing
  else. The same happens for "What color is the sky?" and "Give me three
  words that rhyme with cat.".
- **Answers, sampled:** the plain-text question ends at once in 87% of runs
  (first-token probability of `<|im_end|>` 87%). In Chat mode 98.5% of runs
  contain "Paris", but the model often keeps going after the sentence: after
  "The capital of France is Paris." it writes `<|im_end|>` with 45%, "␣It"
  with 32% and "␣Paris" with 19%. Runs that end their turn themselves: 37%
  within 20 new tokens, 89% within 100. The texts below are worded to these
  numbers.
- **System prompts are followed weakly.** Of five tried, only "You always
  answer in French." clearly changed answers ("What color is the sky?" → "La
  lumière est un color."). "Answer with one word only.", "You are a poet.
  Answer in rhyme.", a pirate persona and a renamed assistant were ignored.
- **Stopping already works:** `generation_config.eos_token_id` is 2,
  `<|im_end|>`, which the worker already treats as end of sequence.
- **Attention** (mean over 30 layers × 9 heads, chat run above): the first
  `<|im_start|>` takes 33–41% (the attention sink), the system line 5–10%,
  the other template tokens 9–27%, the user's words 16–24%. The template
  does not crowd out the user's text. Head L23·H4 puts 58% on "France" in
  the question while writing "Paris".
- **Sizes:** the attention matrix uses 20 px per position, so about 39
  positions fit the 928 px card. A chat run with the default 20 new tokens
  has up to 57 positions (1,276 px); the longest possible chat run, 137. The
  export limits (512 positions for attention, 512 prompt tokens for
  embedding rows) are not reached.

## Decisions already made

- **Two formats:** "Chat" and "Continue text". Chat is real-model only.
- **System prompt:** editable, starting with the model's default, with a
  "Reset to default" link. Empty means no system turn at all.
- **Folding:** the first `<|im_start|>` always stays its own token. The
  system turn after it (system prompt tokens + 4) folds, in the token strip,
  the attention matrix and the Edit sizes card. The data keeps every
  position; folding is presentation only.
- **The attention matrix's cells fit the card width** (both formats).
- **The real model is made prominent** (section 6): a "Simulated | Real
  model" switch at the front of the prompt row replaces the checkbox, a
  banner explains the simulation, and the model's status moves to the
  banner's slot and a badge next to the title.
- **Not included:** multi-turn conversations; Chat in the simulated mode.

## 1. Prompt bar and examples

- **The format switch:** a segmented control "Chat | Continue text" after
  the mode switch (section 6), in front of the prompt field (`data-testid="format-switch"`), shown only while
  "Real model" is on. Its state lives in `PromptBar` like the temperature,
  default Chat. With the real model off, every run is Continue text.
- **The system prompt line:** under the prompt row, only in Chat. Label
  "System prompt", a single-line input (`data-testid="system-input"`)
  prefilled with the default, and a "Reset to default" link-button that
  appears when the value differs from the default. The value is kept for the
  session. An empty or whitespace-only value means no system turn.
- **Prompt placeholder:** "Ask a question…" in Chat, "Type a prompt…" in
  Continue text.
- **Glossary link:** an ⓘ after the switch opens the new `chat-template`
  entry.
- **Examples:** `PromptExample` gains `format?: 'chat' | 'text'` (absent =
  text). A text example sets the switch to Continue text before running. The
  new example:
  - id `chat-question`, label "Chat: a question", prompt "What is the capital
    of France?", format chat, default system prompt (the field is reset).
  - hint: "As Continue text this question usually ends at once: the
    model's first token is <|im_end|> in about 9 of 10 runs. In Chat mode the
    template's tokens tell it a user asked and the assistant should reply.
    It answers, and often ends its turn itself with <|im_end|>; with T = 0 it
    always does."
  - The chip runs with the current sampling settings, like the other
    examples; it does not change them.
  - The chip is drawn like the mockup: a small "Chat" tag in front of its
    label.
  - With the real model off, the chip turns it on (the same path as ticking
    the box, including the download) and starts the run once the model is
    ready. If loading fails, the existing error path applies and nothing
    runs.
- The chat example is not a `CuratedExample` of the simulated engine (it has
  no hand-written continuation); the example list passed to `PromptBar` is
  the curated three plus this one.

## 2. Running the model

- **Request:** `GenParams` stays as is. `WorkerRequest`'s `run` and
  `PipelineEngine.run` gain `format: PromptFormat`, where
  `PromptFormat = { kind: 'text' } | { kind: 'chat'; system: string | null }`.
  `null` means no system turn; the default system prompt is passed as its
  text, not as `null`.
- **Building the input:** with a system prompt, the worker calls the
  tokenizer's own `apply_chat_template` with
  `[{role:'system',content:system},{role:'user',content:prompt}]`,
  `add_generation_prompt: true`, `tokenize: false`, and encodes the string
  with `add_special_tokens: false`.
- **No system turn:** the template inserts its default whenever the first
  message is not a system message, so it cannot produce "no system turn".
  That string is built by hand as
  `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`: the
  template's output minus its first turn. A unit test pins it against that.
- **Locating the parts** (`chatSpans(ids, imStart, imEnd)`, a pure function):
  scan the ids for the special tokens. With a system turn: `system` is
  `[1, s)` where `s` is the index of the second `<|im_start|>`; without one,
  `system` is absent. `user` is `[u, e)`: from 3 tokens after the
  `<|im_start|>` that opens the user turn (`<|im_start|>`, `user`, `\n`) to
  the next `<|im_end|>`. Special-token ids come from the tokenizer
  (`<|im_start|>` 1, `<|im_end|>` 2 for this model; looked up, not
  hard-coded). If the scan does not find the expected shape, the run goes on
  without spans and the views show every position unfolded.
- **Too-long prompts:** the budget stays `max_position_embeddings −
  maxNewTokens`. In Chat, when the encoded input exceeds it, tokens are
  removed from the start of the user's words (the `user` span), never from
  the template, and `truncated` is set. If the template alone exceeds the
  budget, the system turn is dropped as well.
- **Stopping:** unchanged; `<|im_end|>` is already an end-of-sequence id and
  is appended to the strip as the last generated token.
- The fallback stock export (no attention outputs) runs Chat the same way.

## 3. Trace and saved runs

- `run-start` gains `format?: PromptFormat` (absent = text). `prompt` stays
  the user's text, not the template string.
- `tokenize` gains `chat?: { system?: [number, number]; user: [number, number] }`
  (half-open position ranges into `tokens`).
- `RunMeta` gains `format?: PromptFormat`; `archiveSeal` copies it from
  `run-start`. Old records and run files lack it and load as Continue text,
  unchanged. `RUN_FILE_VERSION` stays 1 (the fields are optional).
- The run shelf: a chat run's chip label starts with "Chat · ", and its
  title adds the system prompt ("system: default", "system: none" or the
  text, cut to 60 characters).
- A selector `chatLayout(events)` returns the spans of the run (or
  `undefined`), for the views below.

## 4. Display

### Token strip (`TokenStream`)
- Prompt tokens inside `chat.system` fold into one chip after the first
  `<|im_start|>`: "system prompt · {n} tokens ▸" (`data-testid="system-fold"`),
  a button that opens the folded tokens in place (and "▾" closes them).
  Open/closed is view state, closed by default, shared with nothing.
- Template tokens (prompt positions outside `chat.user`, including the
  folded ones when open) get `token-template` (dashed, grey). Special tokens
  (`<|im_start|>`, `<|im_end|>`) get `token-special` (dashed indigo,
  monospace) and show their literal text.
- A generated `<|im_end|>` gets `token-end` (vermillion), title "end of turn:
  the model stopped here".
- Colours and line styles as in the mockup's legend, with the app's tokens
  (`--ai-soft`, `--line-strong`, `--shu`, `--shu-wash`).

### Tokenizer stage
- The band's summary for a chat run: "{n} tokens · {k} yours" (k = user span
  length); Continue text unchanged.
- `TokenizerDetail` for a chat run: one sentence before the chips: "In Chat
  mode the prompt is wrapped in the model's chat template: {n − k} of the
  {n} tokens were added by the template." with "chat template" as a `Term`.
  Template chips get the dashed style; the system turn folds as in the
  strip.

### Attention matrix (`AttentionHeatmap`, Layers → Attention)
- **Fold:** with spans present and the toggle off, the rows of `chat.system`
  are hidden and its columns become one column after column 0, labelled
  "system prompt ({n})", whose cell value is the sum of the row's weights on
  those positions (dashed outline). Row and column 0 stay. Hover and
  readout name it ("Paris → system prompt (20): 6%"). A checkbox "Show system
  prompt" (`data-testid="attn-show-system"`) above the matrix opens the full
  matrix. Without spans there is no checkbox and no fold. The fold is
  computed by a pure function `foldMatrix(matrix, span)` returning the
  matrix, row positions and column groups.
- **Fit to width:** the cell size is
  `clamp(floor((width − LABEL_W − RIGHT_PAD) / columns), 8, 20)`, where
  `width` is the card's content width (measured with a `ResizeObserver`;
  20 px before the first measurement and in jsdom). Short runs keep 20 px.
  Below 8 px the matrix scrolls sideways as today. Below 14 px, only every
  `ceil(14 / cell)`-th row and column label is drawn, and the hovered row and
  column always get theirs. The slanted-label padding scales with the cell.
- **Hover:** one outline rectangle drawn on top, positioned from hover
  state; the cells no longer depend on hover, so hovering does not re-render
  them (cells in a memoised child).
- **Head hint:** for a chat run whose shown head is labelled `induction`,
  the hint adds: "In Chat mode the template's markers (<|im_start|>, ↵,
  <|im_end|>) repeat, and those repeats count too."
- **Compare view:** each side folds its own chat run the same way (no
  toggle there).

### Edit sizes (`EditSizeCard`)
- The heatmap rows of `chat.system` fold into one row, "system prompt ({n})
  ▸", that is not selectable and shows nothing (each position's edit is its
  own; a sum would mean nothing). Clicking it opens the rows, as the strip's
  chip does; the card has its own open/closed state. The 64-row cap applies
  after folding. The "showing the last … of …" note counts positions.

### Unchanged
- Embedding similarity (last 24 prompt tokens), head grid thumbnails, the
  votes, neuron and step-by-step cards (one selected token).

## 5. Glossary

New entry `chat-template` ("Chat template"), sections:
- **what** — the model was tuned on conversations in one format; every chat
  assistant wraps your message in it; the tokens shown dashed are that
  wrapping.
- **system-prompt** — the system turn, the default line, what an empty field
  does, and how weakly a 135M model follows it (the French example).
- **special-tokens** — `<|im_start|>` and `<|im_end|>` are single vocabulary
  entries that typed text never produces.
- **continue-text** — why a question as plain text usually ends at once
  (about 9 of 10 runs at the default settings, every time with T = 0).
- **sink** — why the first `<|im_start|>` takes the attention sink role.

Linked from the format switch's ⓘ, the fold chip's title, and
`TokenizerDetail`. The `token` entry gets one sentence pointing to it.

## 6. Making the real model hard to miss

Chat mode, attention, the MLP cards and the neuron data all need the real
model, which today hides behind a small checkbox at the end of the settings
row. Nothing here downloads without a click.

- **The mode switch:** a segmented control "Simulated | Real model · 250 MB"
  (`data-testid="mode-switch"`) at the front of the prompt row replaces the
  checkbox (`mode-toggle`). The format switch follows it when the real model
  is on. The checkbox's tooltip text moves to the switch's real option.
- **The banner** (`data-testid="sim-banner"`), in a slot under the example
  chips, while the mode is simulated and the banner has not been closed:
  "**You are watching a simulation.** Its numbers are made up to show the
  shape of the pipeline. Load the real model to see what SmolLM2 actually
  computes: real attention, the MLP inside each layer, and answers in Chat
  mode. 250 MB, downloaded once, runs entirely in your browser." with a
  "Load the real model" button and a close button (×, label "Close").
  Vermillion wash as in the mockup (`--shu-soft` border, `--shu-wash`
  ground).
- **The title badge** (`data-testid="mode-badge"`) next to "Tsumugi":
  "Simulated" (vermillion outline) while simulated, also after the banner is
  closed; "Loading…" while loading; "Real model · {device}{ · attn}" when
  ready (replacing today's device chip in the settings row).
- **Model status in the banner's slot:** while loading, after any "Load the
  real model" click (banner, switch or a card's button), the slot shows
  "Loading the real model: {file} · {percent} %" with a thin progress bar,
  even if the banner was closed. A load error shows there too, with the
  existing "Continue in Simulated mode" button. When the model is ready, the
  slot disappears. `ModelStatus` moves from the settings row into this slot;
  the settings row keeps T, top-k and max.
- **Buttons where the simulation stops:** the "Real model only" notes in the
  Layers detail (MLP, neuron and step-by-step cards) get a "Load the real
  model" button, and the simulated-attention note gets a "Load the real
  model" link. They call one shared action (`requestRealModel` in a small
  store the app subscribes to), not props passed through every card. During
  a load they are disabled.
- **Remembered choices** (`localStorage`, every access in try/catch; without
  storage the app behaves as on a first visit):
  - `tsumugi.simBannerClosed`: set by the close button.
  - `tsumugi.realModelUsed`: set when the real model becomes ready.
  - On start, when `realModelUsed` is set **and** the model's files are in
    the browser cache, the app switches to real mode by itself; the slot
    shows "Loading the real model from the browser's cache…" until ready.
    The cache check (`realModelCached()`) opens the Cache Storage
    `transformers-cache` (transformers.js `env.cacheKey`) and looks for the
    model file of the pinned revision. If it is missing (cache cleared), the
    app starts simulated and shows the banner again, even if it was closed
    before, without downloading anything.
- **The simulated attention note** reads "Illustrative pattern (simulated):
  the real model would show the weights it computed." (today's text claims
  the browser model exposes no attention, which has not been true since the
  attention export).

## Testing

Unit (vitest):
- `chatSpans`: default system, custom system, no system, malformed ids (no
  spans).
- The worker's input string: equals the template's output for default and
  custom system prompts (template string checked in a fixture, since the
  test runner has no tokenizer download); the no-system string equals it
  minus the first turn.
- Truncation in Chat removes user tokens from the start and never template
  tokens; template longer than the budget drops the system turn.
- Trace and archive: `format` reaches `RunMeta`; an old record without it
  loads as text; shelf label and title.
- `PromptBar`: switch hidden in simulated mode; system line only in Chat;
  reset link; text examples set Continue text; the chat chip with the real
  model off requests real mode and runs after ready.
- `TokenStream` and `TokenizerDetail`: fold chip count, open/close, classes
  for template, special and end tokens; Continue text unchanged.
- `foldMatrix`: summed column, hidden rows, column 0 kept, no spans → same
  matrix.
- `AttentionHeatmap`: cell size 20 for short runs, shrinks with columns,
  floor 8; labels thinned below 14 and kept for the hovered row/column;
  hovering does not re-render cells; toggle shows the full matrix.
- `EditSizeCard`: folded row, open/close, cap after folding.
- Mode switch and banner: banner shown on a first visit in simulated mode,
  close remembered, hidden in real mode; badge texts for simulated, loading
  and ready; progress and error in the banner slot even after closing; card
  buttons call `requestRealModel` and are disabled while loading.
- Start-up: `realModelUsed` with cached files → real mode without a click;
  without cached files → simulated with the banner; storage that throws →
  first-visit behaviour. `realModelCached` against a fake Cache Storage.
- Existing tests that tick `mode-toggle` or read `device-chip` move to the
  mode switch and the badge.

Browser (Playwright, real model): with T set to 0 (greedy, so the run is
deterministic), the chat example produces a strip with the fold chip, an
answer containing "Paris" and a final generated `<|im_end|>`, and the
attention matrix fits the card width. This also confirms the greedy findings
on the browser's 8-bit model, not only on fp32. Before calling it done, compare
screenshots of the strip, prompt bar and matrix with the mockups.

## Out of scope

- Multi-turn conversations and a chat transcript view.
- Chat in the simulated mode.
- A different default system prompt, or a picker of preset system prompts.
- Folding in the embedding similarity matrix.

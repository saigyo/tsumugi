# User-facing seed: continue a run, share a run

## Intent

Real-model runs already draw their sampling randomness from a per-run seed
(`run-start.seed`, `cycleDraw(seed, cycle)` in `src/engine/prng.ts`), added for
knockout runs. The draw at each cycle depends only on the seed and the cycle
number, so the same prompt, settings and seed give the same tokens. This
feature puts that in the user's hands for two reasons the user gave:

1. **Continue a cut-off run.** An interesting answer often stops because the
   max-tokens limit was too low. One click should rerun it with a higher limit
   and the same seed, so the text is identical up to the old end and then goes on.
2. **Share a run.** Send someone a link that opens the app with the prompt,
   the settings and the seed filled in, so they can reproduce the result.

Success: the rerun's first N tokens match the original exactly on the same
machine, and a shared link fills every setting needed to replay the run.

## Non-goals

- No seed for simulated runs. The prompt alone already decides them
  (`seedFromTokens` in `SimulatedEngine`). The seed field, the link and the
  continue button are for real-model runs only.
- No guarantee of identical output on another computer (see Limits).
- No server, no short-link service: the link carries everything itself.
- No change to the run file format beyond what already exists. `seed` and
  `revision` are already in `run-start`.

## Behaviour

### Seed field (prompt bar)

- A "Seed" field joins Temperature, Top-k and Max new tokens in the config
  row. It is shown in real mode only, like the format switch.
- It is empty by default. Empty means a new random seed for each run, as now.
- An integer from 0 to 4294967295 fixes the seed for every Generate until the
  field is cleared. Clearing it is valid and means random again. Negative
  numbers, decimals and values above 4294967295 are rejected, not clamped:
  the field keeps its previous value (my call).
- It has an ⓘ term link to a new `seed` section of the Generation settings
  glossary entry, with the tip "Same seed, prompt and settings give the same text."
- After a run with a random seed, the field stays empty (my call). The run's
  seed is shown in the run-end detail instead.
- The example chips keep their current behaviour and ignore the seed field
  (my call: they are demos).

### Engine

- `RunOptions` gains `seed?: number`. The worker uses
  `options.seed ?? knockout?.seed ?? random`. Simulated runs ignore it.
- Knockout runs keep passing their source run's seed as now.

### Run-end detail (`RunEndDetail.tsx`)

For a real-model run whose `run-start` has a `seed`:

- The settings line shows the seed: `(T=0.8, top-k 10, max 20, seed 123456789)`.
- **Continue button**, only when the run ended with `max-tokens` and its
  `maxNewTokens` is below 100: "Run again with N tokens", where
  N = min(100, 2 × maxNewTokens). It starts a new run with the same prompt,
  format (including system prompt), temperature, top-k and seed, and
  maxNewTokens = N. The new run joins the shelf as a normal run; the original
  stays. Compare does not open. The prompt bar's fields are not changed (my
  call: the button is a one-off action, like the knockout run).
- **Copy link button**: copies the share link (below) to the clipboard and
  shows "Link copied" briefly. On clipboard failure, show the link in a
  selectable text field instead.
- Continue always uses the run's own seed, never the prompt bar's seed field.
- Continue needs the same model the run was made with: the real model loaded,
  with the run's `modelId` and `revision`. Otherwise the button is disabled
  and its title says "Load model version vX to continue this run" (my call).
- Continue is hidden when the original run's prompt was truncated
  (`tokenize.truncated`), because a higher limit shrinks the prompt budget
  and the text could differ (my call). A rerun whose prompt plus N exceeds 512
  tokens loses the attention views, as any such run does; the tokens still
  match, so this is accepted.
- Both buttons are disabled while something is generating or measuring.
- Runs recorded before seeded sampling (no `seed`), simulated runs, aborted
  and errored runs: no seed text, no buttons. Knockout runs: no buttons (my
  call: replaying one needs the removed layer, which a link does not carry).

### Share link

- Format: the current page's address (`location.origin + location.pathname`)
  with the settings in the hash fragment, so they are not
  sent to the web server, e.g.
  `https://saigyo.github.io/tsumugi/#run=1&p=<prompt>&f=chat&s=<system>&t=0.8&k=10&n=20&seed=123456789&rev=v7`.
  `f` is `chat` or `text`; `s` appears only for chat. Values are URL-encoded.
  `rev` is the run's model revision (absent for the stock fallback model).
- Opening a link: the app fills the prompt, format, system prompt, temperature,
  top-k, max tokens and seed fields, and switches to the real model (which
  loads it, as the Real model button does). It does not generate; the person
  presses Generate.
- The link is read once, when the app starts; it fills the fields and
  replaces whatever they held (my call; no `hashchange` listener).
- After reading it, the app removes the fragment from the address bar
  (`history.replaceState`), so a reload starts clean (my call).
- Out-of-range or malformed values fall back to the field's default; a link
  with no prompt or no seed fills what it can.
- Model check: the link's `rev` (absent means the stock fallback model) is
  compared with the loaded model's revision (absent when the stock export
  loaded). If they differ, a note under the prompt bar says:
  "This link was made with model version vX; the loaded model is vY, so the
  text may differ." Dismissed by the next Generate.

## Limits (to state in the glossary section)

- A different graphics card or browser can round numbers slightly differently.
  When two tokens are almost tied, that can change a pick, so a shared run
  usually matches but is not guaranteed to.
- A link made with one model version and opened after a newer one is
  published runs on the newer model and can give different text.

## Left to the builder

- How the prompt bar receives values from a link (lifted state, an initial-
  values prop, or a small store).
- Where the parsing and building of the link lives; it should be one module
  with round-trip tests.
- Exact styling of the seed field and the two buttons, following the existing
  config row and detail panel styles.
- README: a short "Continuing and sharing a run" section; the screenshot
  script needs no new scene.

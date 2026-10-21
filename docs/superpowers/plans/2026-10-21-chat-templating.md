# Chat Templating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the real model either as "Continue text" (today) or as "Chat" through SmolLM2's own chat template with an editable system prompt, show the template's tokens with the system turn folded, fit the attention matrix to the card, and make the real model hard to miss.

**Architecture:** A pure module `src/engine/chatTemplate.ts` owns the prompt format, the default system prompt, locating the system and user spans in the token ids, and truncation. The worker applies the template and emits the spans on the `tokenize` event; `run-start` records the format; the archive carries it. The views read the spans through one selector (`chatLayout`) and fold the system turn with shared helpers (`src/viz/chatFold.ts`, `foldMatrix`). A small zustand store lets any card ask the app to load the real model.

**Tech Stack:** React 19, TypeScript, zustand, vitest + Testing Library (jsdom), Playwright, transformers.js (`@huggingface/transformers`).

**Spec:** `docs/superpowers/specs/2026-10-21-chat-templating-design.md`

## Global Constraints

- Branch `feature/chat-template` (exists, holds the spec). Never commit to main, never push, never merge without the user's go-ahead. Integration later is rebase + fast-forward only, never a merge commit.
- TDD: write the test, watch it fail for the right reason, implement, watch it pass.
- Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Checks before each commit: `npx vitest run <changed test files>`; before the final review: `npm test`, `npx tsc -b`, `npm run lint` (2 known warnings are accepted, no new ones), `npm run build`, `npx playwright test`.
- Copy rules from the spec, verbatim:
  - Default system prompt: `You are a helpful AI assistant named SmolLM, trained by Hugging Face`
  - Fold chip: `system prompt · {n} tokens ▸` (open: `system prompt · {n} tokens ▾`)
  - Banner: "**You are watching a simulation.** Its numbers are made up to show the shape of the pipeline. Load the real model to see what SmolLM2 actually computes: real attention, the MLP inside each layer, and answers in Chat mode. 250 MB, downloaded once, runs entirely in your browser."
  - Chat example hint: "As Continue text this question usually ends at once: the model's first token is <|im_end|> in about 9 of 10 runs. In Chat mode the template's tokens tell it a user asked and the assistant should reply. It answers, and often ends its turn itself with <|im_end|>; with T = 0 it always does."
  - Simulated attention note: "Illustrative pattern (simulated): the real model would show the weights it computed."
- Colours only from the CSS tokens in `src/index.css` `:root` (`--ai`, `--ai-soft`, `--ai-wash`, `--line-strong`, `--ink-faint`, `--shu`, `--shu-soft`, `--shu-wash`, …).
- Old traces, saved runs and run files without the new fields must render exactly as before.
- **Ruling (plan vs spec):** the spec's Playwright test "with the real model" cannot run in CI (CI never downloads the model; all e2e tests use imported fixture runs). CI gets an imported chat-fixture run (Task 11); the greedy real-model check (T = 0, "Paris", final `<|im_end|>`) is a manual verification step in Task 11, done locally before reporting. Cost if wrong: the greedy claim is not re-checked on every CI run.
- **Ruling:** the spec says the matrix's slanted-label padding scales with the cell. Labels keep their 10 px font, so the padding they need does not shrink with the cell; it stays fixed and label thinning prevents overlap. Cost if wrong: a few pixels of extra space above small matrices.

## Review Focus

1. A prompt or system prompt that contains the literal text `<|im_start|>` or `<|im_end|>`: the tokenizer turns it into special tokens, so the span scan sees an unexpected shape. Expected: the run works, spans are absent or still inside the token range, nothing crashes; the strip shows every token unfolded. (Test in Task 1.)
2. Switching the mode switch to Simulated while the format switch says Chat: the run must be Continue text, not a chat run in the simulator. (Test in Task 9.)
3. A whitespace-only system prompt: no system turn, no fold chip, the "{n} tokens · {k} yours" summary still right. (Tests in Tasks 1 and 5.)
4. A pinned attention head whose matrix is longer or shorter than the system span, or a cycle-0 matrix: the fold must never index past a row. (Test in Task 6.)
5. A saved run from before this change (no `format`, no `chat`): strip, tokenizer detail, matrix, edit sizes and shelf look exactly as before, and a run file with a malformed `format` loads as text. (Tests in Tasks 4, 5, 6 and 8.)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/trace/types.ts` (modify) | `PromptFormat`, `ChatSpans`; `run-start.format?`, `tokenize.chat?` |
| `src/engine/chatTemplate.ts` (create) | default system prompt, `systemOf`, `noSystemChat`, `chatSpans`, `fitChatToBudget`, `isSpecialToken`, `TEXT_FORMAT` |
| `src/glossary/entries/chat-template.tsx` (create) | the new glossary entry |
| `src/engine/transformers/{protocol,TransformersEngine,worker}.ts`, `src/engine/types.ts` (modify) | carry the format to the worker, build the chat input, emit spans |
| `src/app/runsStore.ts`, `src/app/runFiles.ts`, `src/app/RunShelf.tsx`, `src/App.tsx` (modify) | archive the format; shelf label and title |
| `src/viz/selectors.ts` (modify) | `chatLayout(events)` |
| `src/test/fixtures.ts` (modify) | `chat` option: the real 37-token chat prompt and answer |
| `src/viz/chatFold.ts` (create) | `foldItems`, `tokenRole` for strip, tokenizer detail and edit sizes |
| `src/viz/TokenStream.tsx`, `src/viz/details/TokenizerDetail.tsx`, `src/viz/PipelineBand.tsx` (modify) | fold chip, token styles, summary |
| `src/viz/foldMatrix.ts` (create) | fold a ragged attention matrix |
| `src/viz/heatmapSizing.ts` (create) | `cellSize`, `labelStep` |
| `src/viz/AttentionHeatmap.tsx` (modify) | fold + toggle, fit to width, hover outline, chat hint, sim note |
| `src/viz/EditSizeCard.tsx` (modify) | folded system rows |
| `src/app/PromptBar.tsx`, `src/app/chatExample.ts` (create) | mode switch, format switch, system line, chat example |
| `src/app/realModel.ts` (create) | `useRealModel` store, prefs in localStorage, `realModelCached` |
| `src/app/LoadRealButton.tsx`, `src/app/ModelSlot.tsx`, `src/app/ModeBadge.tsx` (create) | the card buttons, the banner slot, the title badge |
| `src/app/ModelStatus.tsx` (delete in Task 10, replaced by `ModelSlot`) | |
| `e2e/chat.spec.ts` (create) | imported chat run in the browser |

---

### Task 1: Prompt format types and the chat template module

**Files:**
- Modify: `src/trace/types.ts`
- Create: `src/engine/chatTemplate.ts`
- Test: `src/engine/chatTemplate.test.ts`

**Interfaces:**
- Produces (in `src/trace/types.ts`):
  - `export type PromptFormat = { kind: 'text' } | { kind: 'chat'; system: string | null }`
  - `export interface ChatSpans { system?: [number, number]; user: [number, number] }` (half-open position ranges)
- Produces (in `src/engine/chatTemplate.ts`):
  - `DEFAULT_SYSTEM_PROMPT: string`, `TEXT_FORMAT: PromptFormat`
  - `systemOf(raw: string): string | null`
  - `noSystemChat(prompt: string): string`
  - `chatSpans(ids: number[], imStart: number, imEnd: number): ChatSpans | undefined`
  - `fitChatToBudget(ids: number[], spans: ChatSpans, budget: number): { ids: number[]; spans: ChatSpans; truncated: boolean }`
  - `isSpecialToken(text: string): boolean`

- [ ] **Step 1: Add the types**

In `src/trace/types.ts`, after the `Mode` line, add:

```ts
// How a real run's prompt reached the model: as plain text to continue, or
// wrapped in the model's chat template. system null = no system turn at all.
export type PromptFormat = { kind: 'text' } | { kind: 'chat'; system: string | null }

// Chat runs: half-open position ranges into the tokenize event's tokens. The
// system turn runs from "system" to the "\n" after its <|im_end|>; the user
// span is the user's own words.
export interface ChatSpans { system?: [number, number]; user: [number, number] }
```

- [ ] **Step 2: Write the failing tests**

Create `src/engine/chatTemplate.test.ts`:

```ts
import { expect, test } from 'vitest'
import { chatSpans, DEFAULT_SYSTEM_PROMPT, fitChatToBudget, isSpecialToken, noSystemChat, systemOf } from './chatTemplate'

const S = 1, E = 2   // <|im_start|>, <|im_end|>
// The real ids of the default chat prompt for "What is the capital of France?"
// (SmolLM2-135M-Instruct tokenizer): system turn [1, 21), user words [24, 31).
const DEFAULT_IDS = [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411, 407,
  19712, 8182, 2, 198, 1, 4093, 198, 1780, 314, 260, 3575, 282, 4649, 47, 2, 198, 1, 520, 9531, 198]
// no system turn: <|im_start|> user \n Hi <|im_end|> \n <|im_start|> ass istant \n
const NO_SYSTEM_IDS = [S, 4093, 198, 9000, E, 198, S, 520, 9531, 198]

test('the default system prompt is the model template’s own line', () => {
  expect(DEFAULT_SYSTEM_PROMPT).toBe('You are a helpful AI assistant named SmolLM, trained by Hugging Face')
})

test('an empty or whitespace-only system prompt means no system turn', () => {
  expect(systemOf('')).toBeNull()
  expect(systemOf('   ')).toBeNull()
  expect(systemOf('You always answer in French.')).toBe('You always answer in French.')
})

test('without a system turn the string is the template’s output minus its first turn', () => {
  // the template with a system message, as produced by apply_chat_template for "Hi"
  const withSystem = '<|im_start|>system\nX<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n'
  expect(noSystemChat('Hi')).toBe(withSystem.slice(withSystem.indexOf('<|im_start|>user')))
})

test('spans of the default chat prompt: system [1, 21), user [24, 31)', () => {
  expect(chatSpans(DEFAULT_IDS, S, E)).toEqual({ system: [1, 21], user: [24, 31] })
})

test('spans without a system turn have only the user part', () => {
  expect(chatSpans(NO_SYSTEM_IDS, S, E)).toEqual({ user: [3, 4] })
})

test('an unexpected shape gives no spans', () => {
  expect(chatSpans([], S, E)).toBeUndefined()
  expect(chatSpans([4093, 198], S, E)).toBeUndefined()                          // no <|im_start|>
  expect(chatSpans([9, S, 4093, 198, 7, E, 198, S, 520], S, E)).toBeUndefined()  // not at position 0
  expect(chatSpans([S, 4093, 198, 7, 198, S, 520], S, E)).toBeUndefined()        // user turn never closed
})

test('special tokens typed into the prompt never crash the scan and never give spans outside the ids', () => {
  // user typed "<|im_start|>": a fourth <|im_start|>
  const extraStart = [...DEFAULT_IDS.slice(0, 26), S, ...DEFAULT_IDS.slice(26)]
  expect(chatSpans(extraStart, S, E)).toBeUndefined()
  // user typed "<|im_end|>": the user span ends early but stays inside the ids
  const extraEnd = [...DEFAULT_IDS.slice(0, 26), E, ...DEFAULT_IDS.slice(26)]
  const spans = chatSpans(extraEnd, S, E)
  if (spans) {
    expect(spans.user[0]).toBeLessThanOrEqual(spans.user[1])
    expect(spans.user[1]).toBeLessThanOrEqual(extraEnd.length)
  }
})

test('within budget nothing is cut', () => {
  const spans = chatSpans(DEFAULT_IDS, S, E)!
  expect(fitChatToBudget(DEFAULT_IDS, spans, 100)).toEqual({ ids: DEFAULT_IDS, spans, truncated: false })
})

test('over budget, the start of the user’s words goes first and the template stays', () => {
  const spans = chatSpans(DEFAULT_IDS, S, E)!
  const r = fitChatToBudget(DEFAULT_IDS, spans, DEFAULT_IDS.length - 2)
  expect(r.truncated).toBe(true)
  expect(r.ids).toHaveLength(DEFAULT_IDS.length - 2)
  expect(r.ids.slice(0, 24)).toEqual(DEFAULT_IDS.slice(0, 24))   // template up to the user words intact
  expect(r.ids.slice(24, 29)).toEqual(DEFAULT_IDS.slice(26, 31)) // "What", " is" removed
  expect(r.ids.slice(29)).toEqual(DEFAULT_IDS.slice(31))         // closing template intact
  expect(r.spans).toEqual({ system: [1, 21], user: [24, 29] })
})

test('when the user’s words are not enough, the system turn is dropped too', () => {
  const spans = chatSpans(DEFAULT_IDS, S, E)!
  const r = fitChatToBudget(DEFAULT_IDS, spans, 12)
  expect(r.truncated).toBe(true)
  expect(r.ids[0]).toBe(S)
  expect(r.ids.slice(0, 3)).toEqual([S, 4093, 198])   // the user turn now opens the prompt
  expect(r.spans.system).toBeUndefined()
  expect(r.spans.user).toEqual([3, 3])
})

test('special tokens are recognised by their literal text', () => {
  expect(isSpecialToken('<|im_start|>')).toBe(true)
  expect(isSpecialToken('<|im_end|>')).toBe(true)
  expect(isSpecialToken('user')).toBe(false)
  expect(isSpecialToken(' <|im_end|>')).toBe(false)
})
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `npx vitest run src/engine/chatTemplate.test.ts`
Expected: FAIL, "Failed to resolve import ./chatTemplate".

- [ ] **Step 4: Implement**

Create `src/engine/chatTemplate.ts`:

```ts
import type { ChatSpans, PromptFormat } from '../trace/types'

// The system turn SmolLM2's chat template inserts when the conversation has
// no system message of its own.
export const DEFAULT_SYSTEM_PROMPT = 'You are a helpful AI assistant named SmolLM, trained by Hugging Face'

export const TEXT_FORMAT: PromptFormat = { kind: 'text' }

// The system prompt field: empty or whitespace-only means no system turn.
export const systemOf = (raw: string): string | null => (raw.trim() === '' ? null : raw)

// The template inserts its default system turn whenever the first message is
// not a system message, so "no system turn" cannot come from the template:
// this is its output with the first turn removed.
export const noSystemChat = (prompt: string): string =>
  `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`

// Locates the system turn and the user's words in a chat prompt's ids. The
// expected shape: <|im_start|> at 0, a system turn (optional), the user turn
// "<|im_start|> user \n … <|im_end|> \n", and the generation prompt
// "<|im_start|> ass istant \n". Anything else gives undefined, and the views
// show every position unfolded.
export function chatSpans(ids: number[], imStart: number, imEnd: number): ChatSpans | undefined {
  const starts = ids.flatMap((id, i) => (id === imStart ? [i] : []))
  if (starts[0] !== 0 || (starts.length !== 2 && starts.length !== 3)) return undefined
  const hasSystem = starts.length === 3
  const userTurn = hasSystem ? starts[1] : starts[0]
  const u = userTurn + 3                       // after <|im_start|>, "user", "\n"
  const e = ids.indexOf(imEnd, u)
  if (e < 0 || e > starts[starts.length - 1]) return undefined
  return hasSystem ? { system: [1, starts[1]], user: [u, e] } : { user: [u, e] }
}

// Fits a chat prompt into the position budget: the start of the user's words
// goes first, then (if still too long) the whole system turn. The template's
// markers are never cut.
export function fitChatToBudget(ids: number[], spans: ChatSpans, budget: number):
  { ids: number[]; spans: ChatSpans; truncated: boolean } {
  if (ids.length <= budget) return { ids, spans, truncated: false }
  let over = ids.length - budget
  const [u, e] = spans.user
  const cut = Math.min(over, e - u)
  let out = [...ids.slice(0, u), ...ids.slice(u + cut)]
  let next: ChatSpans = { ...spans, user: [u, e - cut] }
  over -= cut
  if (over > 0 && next.system) {
    const turnEnd = next.system[1]             // the system turn is [0, turnEnd), <|im_start|> included
    out = out.slice(turnEnd)
    next = { user: [next.user[0] - turnEnd, next.user[1] - turnEnd] }
  }
  return { ids: out, spans: next, truncated: true }
}

// <|im_start|>, <|im_end|> and their kin: one vocabulary entry each, never typed.
export const isSpecialToken = (text: string): boolean => /^<\|[a-z_]+\|>$/.test(text)
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `npx vitest run src/engine/chatTemplate.test.ts`
Expected: PASS (11 tests).

- [ ] **Step 6: Commit**

```bash
git add src/trace/types.ts src/engine/chatTemplate.ts src/engine/chatTemplate.test.ts
git commit -m "feat(chat): prompt format types; locate system and user spans; truncate without cutting the template

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The "Chat template" glossary entry

**Files:**
- Create: `src/glossary/entries/chat-template.tsx`
- Modify: `src/glossary/types.ts` (`GLOSSARY_IDS`), `src/glossary/registry.ts` (`ENTRIES`), `src/glossary/entries/token.tsx`
- Test: `src/glossary/registry.test.tsx` (existing tests cover body, summary, ids); add one test

**Interfaces:**
- Produces: glossary id `'chat-template'` with sections `what`, `system-prompt`, `special-tokens`, `continue-text`, `sink`. Later tasks use `<Term id="chat-template">`.

- [ ] **Step 1: Write the failing test**

Append to `src/glossary/registry.test.tsx`:

```ts
test('the chat template entry has its five sections and sits in the Tokenizer group', () => {
  expect(sectionsOf('chat-template')).toEqual(['what', 'system-prompt', 'special-tokens', 'continue-text', 'sink'])
  expect(getEntry('chat-template')?.stage).toBe('tokenize')
})
```

- [ ] **Step 2: Run it to see it fail**

Run: `npx vitest run src/glossary/registry.test.tsx`
Expected: FAIL, the new test gets `[]` (no entry).

- [ ] **Step 3: Implement**

In `src/glossary/types.ts`, add `'chat-template'` to `GLOSSARY_IDS` right after `'token'`.

Create `src/glossary/entries/chat-template.tsx`:

```tsx
import { Section } from '../Section'
import { Term } from '../Term'
import type { GlossaryEntry } from '../types'

export const chatTemplate: GlossaryEntry = {
  id: 'chat-template',
  title: 'Chat template',
  summary: 'The fixed wrapping of special tokens and role names that turns your message into the conversation format the model was tuned on.',
  stage: 'tokenize',
  related: ['token', 'attention', 'modes'],
  body: () => (
    <>
      <Section id="what" title="What it is">
        <p>SmolLM2-135M-Instruct was tuned on conversations written in one format. Every chat assistant wraps your
          message in that format before the model sees it:</p>
        <p className="gl-example"><code>{'<|im_start|>system↵You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>↵'}</code><br />
          <code>{'<|im_start|>user↵What is the capital of France?<|im_end|>↵'}</code><br />
          <code>{'<|im_start|>assistant↵'}</code></p>
        <p>In Chat mode Tsumugi does the same, with the template that ships with the model. The tokens drawn dashed
          in the token strip are that wrapping; your own words are drawn solid.</p>
      </Section>
      <Section id="system-prompt" title="The system prompt">
        <p>The first turn tells the model who it is. When no system message is given, the template inserts its
          default line. You can replace it, or empty the field to leave the system turn out altogether.</p>
        <p className="gl-aside">A model with 135 million parameters follows a system prompt only weakly. “You always
          answer in French.” changes some answers; “Answer with one word only.” is mostly ignored.</p>
      </Section>
      <Section id="special-tokens" title="Special tokens">
        <p><code>{'<|im_start|>'}</code> and <code>{'<|im_end|>'}</code> are single vocabulary entries, one{' '}
          <Term id="token">token</Term> each. Typed text never produces them; only the template does. The model
          ends its answer by writing <code>{'<|im_end|>'}</code> itself.</p>
      </Section>
      <Section id="continue-text" title="Why a question as plain text ends at once">
        <p>In Continue text mode the model reads “What is the capital of France?” as a finished document, a turn
          that is complete. Its most likely next token is <code>{'<|im_end|>'}</code>: in about 9 of 10 runs at the
          default settings, and every time with T = 0. The run ends with nothing else.</p>
      </Section>
      <Section id="sink" title="Why the first token draws so much attention">
        <p>Many heads park surplus <Term id="attention">attention</Term> on the very first token. In Chat mode that
          token is <code>{'<|im_start|>'}</code>, so it takes over the role the first word of your prompt plays in
          Continue text mode. The system line itself gets only a small share.</p>
      </Section>
    </>
  ),
}
```

In `src/glossary/registry.ts`: `import { chatTemplate } from './entries/chat-template'` and insert `chatTemplate` into `ENTRIES` right after `token`.

In `src/glossary/entries/token.tsx`, after the sentence ending "which ends a turn.", add inside the same `<p>`:

```tsx
{' '}In Chat mode the prompt is wrapped in more of these: see <Term id="chat-template">chat template</Term>.
```

and add `import { Term } from '../Term'` at the top.

- [ ] **Step 4: Run the glossary tests**

Run: `npx vitest run src/glossary`
Expected: PASS (every existing registry check, including one-sentence summary and unique ids, plus the new test).

- [ ] **Step 5: Commit**

```bash
git add src/glossary
git commit -m "feat(glossary): chat template entry with system prompt, special tokens, continue-text and sink sections

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Carry the format to the worker; build the chat input; emit spans

**Files:**
- Modify: `src/trace/types.ts` (event fields), `src/engine/types.ts`, `src/engine/transformers/protocol.ts`, `src/engine/transformers/TransformersEngine.ts`, `src/engine/transformers/worker.ts`
- Test: `src/engine/transformers/TransformersEngine.test.ts`

**Interfaces:**
- Consumes: `PromptFormat`, `ChatSpans`, `TEXT_FORMAT`, `noSystemChat`, `chatSpans`, `fitChatToBudget` (Task 1).
- Produces:
  - `run-start` event: `format?: PromptFormat` (set only for chat runs).
  - `tokenize` event: `chat?: ChatSpans`.
  - `PipelineEngine.run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void, format?: PromptFormat): RunHandle`
  - `WorkerRequest` run: `{ type: 'run'; runId: number; prompt: string; params: GenParams; format: PromptFormat }`

- [ ] **Step 1: Write the failing test**

Append to `src/engine/transformers/TransformersEngine.test.ts`:

```ts
test('run sends the prompt format to the worker; plain text when none is given', () => {
  const { worker, engine } = make()
  engine.run('Hi', { temperature: 0, topK: 1, maxNewTokens: 1 }, () => {}, { kind: 'chat', system: null })
  engine.run('Hi', { temperature: 0, topK: 1, maxNewTokens: 1 }, () => {})
  expect(worker.sent[0]).toMatchObject({ type: 'run', format: { kind: 'chat', system: null } })
  expect(worker.sent[1]).toMatchObject({ type: 'run', format: { kind: 'text' } })
})
```

- [ ] **Step 2: Run it to see it fail**

Run: `npx vitest run src/engine/transformers/TransformersEngine.test.ts`
Expected: FAIL (TypeScript passes in vitest, but `format` is missing from the posted message).

- [ ] **Step 3: Implement the types and the engine**

`src/trace/types.ts`, the two event variants become:

```ts
  | { type: 'run-start'; prompt: string; mode: Mode; modelId: string; params: GenParams; vocabSize?: number
      // chat runs only; absent = Continue text (and every run recorded before chat mode)
      format?: PromptFormat }
  | { type: 'tokenize'; tokens: TokenInfo[]; truncated?: boolean; chat?: ChatSpans }
```

`src/engine/types.ts`:

```ts
import type { GenParams, PromptFormat, TraceEvent } from '../trace/types'
...
  run(prompt: string, params: GenParams, emit: (e: TraceEvent) => void, format?: PromptFormat): RunHandle
```

`src/engine/transformers/protocol.ts`: import `PromptFormat` and change the run request to
`| { type: 'run'; runId: number; prompt: string; params: GenParams; format: PromptFormat }`.

`src/engine/transformers/TransformersEngine.ts`: `run(prompt, params, emit, format: PromptFormat = TEXT_FORMAT)` (import `TEXT_FORMAT` from `../chatTemplate`, `PromptFormat` from the trace types) and post `{ type: 'run', runId, prompt, params, format }`.

`SimulatedEngine.run` keeps its three parameters (TypeScript accepts the shorter signature); the simulator always runs plain text.

- [ ] **Step 4: Implement the worker**

In `src/engine/transformers/worker.ts`:

- Import `chatSpans, fitChatToBudget, noSystemChat` from `../chatTemplate` and `ChatSpans, PromptFormat` from the trace types.
- The message handler: `if (msg.type === 'run') await run(msg.runId, msg.prompt, msg.params, msg.format ?? { kind: 'text' })`.
- `async function run(runId: number, prompt: string, params: GenParams, format: PromptFormat)`.
- `run-start` gains `...(format.kind === 'chat' ? { format } : {})`.
- Replace the block from `let promptIds…` to the `tokenize` emit with:

```ts
  const maxCtx: number = model.config.max_position_embeddings ?? 2048
  const budget = Math.max(1, maxCtx - params.maxNewTokens)
  let promptIds: number[]
  let chat: ChatSpans | undefined
  let truncated = false
  if (format.kind === 'chat') {
    const text: string = format.system === null
      ? noSystemChat(prompt)
      : tokenizer.apply_chat_template(
        [{ role: 'system', content: format.system }, { role: 'user', content: prompt }],
        { tokenize: false, add_generation_prompt: true })
    promptIds = tokenizer.encode(text, { add_special_tokens: false })
    chat = chatSpans(promptIds,
      tokenizer.convert_tokens_to_ids('<|im_start|>'), tokenizer.convert_tokens_to_ids('<|im_end|>'))
    if (chat) {
      ;({ ids: promptIds, spans: chat, truncated } = fitChatToBudget(promptIds, chat, budget))
    }
  } else {
    promptIds = tokenizer.encode(prompt, { add_special_tokens: false })
  }
  if (promptIds.length > budget) { promptIds = promptIds.slice(-budget); truncated = true; chat = undefined }
  emit({ type: 'tokenize', tokens: promptIds.map(tokenInfo),
    ...(truncated ? { truncated } : {}), ...(chat ? { chat } : {}) })
```

(The old `const maxCtx`, `budget` and `truncated` lines above it are removed; the rest of `run` is unchanged.)

- [ ] **Step 5: Run the engine tests and the type check**

Run: `npx vitest run src/engine && npx tsc -b`
Expected: PASS; no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/trace/types.ts src/engine
git commit -m "feat(chat): the worker builds chat prompts with the model's template and emits the system and user spans

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Archive the format; shelf label; `chatLayout`; chat fixture

**Files:**
- Modify: `src/app/runsStore.ts`, `src/app/runFiles.ts`, `src/app/RunShelf.tsx`, `src/App.tsx` (seal), `src/viz/selectors.ts`, `src/test/fixtures.ts`
- Test: `src/app/runFiles.test.ts`, `src/app/RunShelf.test.tsx`, `src/viz/selectors.test.ts`

**Interfaces:**
- Consumes: `PromptFormat`, `ChatSpans`, `run-start.format`, `tokenize.chat` (Tasks 1, 3).
- Produces:
  - `RunMeta.format?: PromptFormat`
  - `chatLayout(events: TraceEvent[]): ChatSpans | undefined` in `src/viz/selectors.ts`
  - `buildFixtureTrace({ chat: true, … })`: prompt = the real 37 tokens (system [1, 21), user [24, 31)), chosen = "The capital of France is Paris." then `<|im_end|>` (id 2), `cycles` defaults to 8, `run-start.format = { kind: 'chat', system: DEFAULT_SYSTEM_PROMPT }`, reason defaults to `'eos'`.
  - Exported constants `CHAT_PROMPT_TOKENS: TokenInfo[]` and `CHAT_ANSWER_TOKENS: TokenInfo[]` from `src/test/fixtures.ts`.

- [ ] **Step 1: Write the failing tests**

`src/viz/selectors.test.ts`, append:

```ts
test('chatLayout returns the run’s spans, or undefined for a plain-text run', () => {
  expect(chatLayout(buildFixtureTrace({ chat: true }))).toEqual({ system: [1, 21], user: [24, 31] })
  expect(chatLayout(buildFixtureTrace())).toBeUndefined()
})
```

(add `chatLayout` to the selectors import and `buildFixtureTrace` from `../test/fixtures` if not imported.)

`src/app/runFiles.test.ts`, append (use the file's existing helper for building a valid file; the meta below extends it):

```ts
test('a chat run file keeps its format; a malformed format is dropped and the run loads as text', () => {
  const events = buildFixtureTrace({ chat: true, mode: 'real' })
  const meta = { prompt: 'What is the capital of France?', params: { temperature: 0, topK: 1, maxNewTokens: 8 },
    mode: 'real', endedAt: 1, reason: 'eos' }
  const good = parseRunFile(JSON.stringify({ formatVersion: 1, events, meta: { ...meta, format: { kind: 'chat', system: null } } }))
  expect(good.ok && good.meta.format).toEqual({ kind: 'chat', system: null })
  const bad = parseRunFile(JSON.stringify({ formatVersion: 1, events, meta: { ...meta, format: { kind: 'chat', system: 3 } } }))
  expect(bad.ok && bad.meta.format).toBeUndefined()
  const old = parseRunFile(JSON.stringify({ formatVersion: 1, events, meta }))
  expect(old.ok && old.meta.format).toBeUndefined()
})
```

`src/app/RunShelf.test.tsx`, append (follow the file's render helper; `makeRunRecord` takes fixture options):

```ts
test('a chat run’s chip says Chat and its title names the system prompt', () => {
  const chat = makeRunRecord(1, { mode: 'real', prompt: 'What is the capital of France?' })
  chat.meta.format = { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' }
  const none = makeRunRecord(2, { mode: 'real', prompt: 'Hi there' })
  none.meta.format = { kind: 'chat', system: null }
  const plain = makeRunRecord(3, { prompt: 'The cat sat' })
  renderShelf({ records: [chat, none, plain] })
  const mains = screen.getAllByTestId('run-chip-main')
  expect(mains[0]).toHaveTextContent('#1 · Chat · What is the')
  expect(mains[0]).toHaveAttribute('title', expect.stringContaining('system: default'))
  expect(mains[1]).toHaveAttribute('title', expect.stringContaining('system: none'))
  expect(mains[2]).toHaveTextContent('#3 · The cat sat')
  expect(mains[2].getAttribute('title')).not.toContain('system:')
})
```

(If `RunShelf.test.tsx` has no `renderShelf` helper, render `<RunShelf … />` with the props the file's other tests use.)

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run src/viz/selectors.test.ts src/app/runFiles.test.ts src/app/RunShelf.test.tsx`
Expected: FAIL (no `chatLayout`, no `chat` fixture option, no format in meta, old label).

- [ ] **Step 3: Implement the fixture**

In `src/test/fixtures.ts`:

```ts
import { DEFAULT_SYSTEM_PROMPT } from '../engine/chatTemplate'

const tk = (pairs: Array<[number, string]>): TokenInfo[] => pairs.map(([id, text]) => ({ id, text }))
// The real chat prompt for "What is the capital of France?" with the default
// system prompt (SmolLM2-135M-Instruct): system turn [1, 21), user words [24, 31).
export const CHAT_PROMPT_TOKENS = tk([[1, '<|im_start|>'], [9690, 'system'], [198, '\n'], [2683, 'You'], [359, ' are'],
  [253, ' a'], [5356, ' helpful'], [5646, ' AI'], [11173, ' assistant'], [3365, ' named'], [3511, ' Sm'], [308, 'ol'],
  [34519, 'LM'], [28, ','], [7018, ' trained'], [411, ' by'], [407, ' H'], [19712, 'ugging'], [8182, ' Face'],
  [2, '<|im_end|>'], [198, '\n'], [1, '<|im_start|>'], [4093, 'user'], [198, '\n'], [1780, 'What'], [314, ' is'],
  [260, ' the'], [3575, ' capital'], [282, ' of'], [4649, ' France'], [47, '?'], [2, '<|im_end|>'], [198, '\n'],
  [1, '<|im_start|>'], [520, 'ass'], [9531, 'istant'], [198, '\n']])
export const CHAT_ANSWER_TOKENS = tk([[504, 'The'], [3575, ' capital'], [282, ' of'], [4649, ' France'], [314, ' is'],
  [7042, ' Paris'], [30, '.'], [2, '<|im_end|>']])
```

Add `chat?: boolean` to `FixtureTraceOpts` (comment: "a chat run: the real 37-token chat prompt, the answer, spans and format"). At the top of `buildFixtureTrace`:

```ts
  const cycles = opts.cycles ?? (opts.chat ? CHAT_ANSWER_TOKENS.length : 2)
  ...
  const promptTokens = opts.promptTokens ?? (opts.chat ? CHAT_PROMPT_TOKENS : [{ id: 10, text: 'The' }, { id: 11, text: ' cat' }])
  const chosenFor = (c: number): TokenInfo =>
    opts.chosen?.[c] ?? (opts.chat ? CHAT_ANSWER_TOKENS[c % CHAT_ANSWER_TOKENS.length]
      : { id: 100 + c, text: DEFAULT_WORDS[c % DEFAULT_WORDS.length] })
```

and the first two events:

```ts
    { type: 'run-start', prompt: opts.prompt ?? (opts.chat ? 'What is the capital of France?' : 'The cat'),
      mode: opts.mode ?? 'sim', modelId: 'fixture',
      params: { temperature, topK: 10, maxNewTokens: cycles }, vocabSize: 49152,
      ...(opts.chat ? { format: { kind: 'chat', system: DEFAULT_SYSTEM_PROMPT } as const } : {}) },
    { type: 'tokenize', tokens: promptTokens, ...(opts.chat ? { chat: { system: [1, 21] as [number, number], user: [24, 31] as [number, number] } } : {}) },
```

and the closing event: `reason: opts.reason ?? (opts.chat ? 'eos' : 'max-tokens')`.

- [ ] **Step 4: Implement the selector, archive, file parsing, shelf and seal**

`src/viz/selectors.ts`:

```ts
// A chat run's system and user spans (from its tokenize event), or undefined
// for a plain-text run, a run recorded before chat mode, or a failed scan.
export function chatLayout(events: TraceEvent[]): ChatSpans | undefined {
  for (const e of events) if (e.type === 'tokenize') return e.chat
  return undefined
}
```

(import `ChatSpans` with the other trace types.)

`src/app/runsStore.ts`, `RunMeta` gains after `modelId`:

```ts
  format?: PromptFormat     // chat runs only; absent = Continue text
```

`src/app/runFiles.ts`, add a validator and copy it into `meta`:

```ts
const formatOf = (f: unknown): PromptFormat | undefined => {
  if (typeof f !== 'object' || f === null) return undefined
  const v = f as { kind?: unknown; system?: unknown }
  if (v.kind === 'text') return { kind: 'text' }
  if (v.kind === 'chat' && (v.system === null || typeof v.system === 'string')) return { kind: 'chat', system: v.system }
  return undefined
}
...
      ...(typeof m.modelId === 'string' ? { modelId: m.modelId } : {}),
      ...(formatOf(m.format) ? { format: formatOf(m.format) } : {}),
```

`src/App.tsx`, in `archiveSeal({ … })` add `...(start.format ? { format: start.format } : {})`.

`src/app/RunShelf.tsx`:

```ts
import { DEFAULT_SYSTEM_PROMPT } from '../engine/chatTemplate'

const isChat = (r: RunRecord) => r.meta.format?.kind === 'chat'
const label = (r: RunRecord) =>
  `#${r.meta.seq} · ${isChat(r) ? 'Chat · ' : ''}${r.meta.prompt.trim().split(/\s+/).slice(0, 3).join(' ')} · T=${r.meta.params.temperature}`
const systemNote = (r: RunRecord): string => {
  const f = r.meta.format
  if (f?.kind !== 'chat') return ''
  if (f.system === null) return ' · system: none'
  if (f.system === DEFAULT_SYSTEM_PROMPT) return ' · system: default'
  return ` · system: ${f.system.length > 60 ? `${f.system.slice(0, 60)}…` : f.system}`
}
```

and the title becomes `` `${r.meta.prompt} — ${r.meta.mode}${systemNote(r)} · ended: ${r.meta.reason}` ``.

- [ ] **Step 5: Run the tests**

Run: `npx vitest run src/viz/selectors.test.ts src/app src/App.test.tsx`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/app src/App.tsx src/viz/selectors.ts src/viz/selectors.test.ts src/test/fixtures.ts
git commit -m "feat(chat): saved runs keep their format; shelf marks chat runs; chatLayout selector and chat fixture

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Token strip, tokenizer detail and stage summary

**Files:**
- Create: `src/viz/chatFold.ts`, `src/viz/chatFold.test.ts`
- Modify: `src/viz/TokenStream.tsx`, `src/viz/details/TokenizerDetail.tsx`, `src/viz/DetailPanel.tsx` (pass `chat`), `src/viz/PipelineBand.tsx`, `src/index.css`
- Test: `src/viz/TokenStream.test.tsx`, `src/viz/PipelineBand.test.tsx`, `src/viz/DetailPanel.test.tsx`

**Interfaces:**
- Consumes: `chatLayout`, `CHAT_PROMPT_TOKENS`, `buildFixtureTrace({ chat: true })` (Task 4), `isSpecialToken` (Task 1), `<Term id="chat-template">` (Task 2).
- Produces (in `src/viz/chatFold.ts`):
  - `type FoldItem = { kind: 'token'; index: number } | { kind: 'fold'; count: number }`
  - `foldItems(length: number, system: [number, number] | undefined, open: boolean): FoldItem[]`
  - `type TokenRole = 'prompt' | 'template' | 'special'`
  - `tokenRole(index: number, text: string, chat: ChatSpans | undefined): TokenRole`
  - `foldLabel(count: number, open: boolean): string` → `system prompt · ${count} tokens ▸|▾`
  - CSS classes `token-template`, `token-special`, `token-end`, `token-fold`.

- [ ] **Step 1: Write the failing tests**

Create `src/viz/chatFold.test.ts`:

```ts
import { expect, test } from 'vitest'
import { foldItems, foldLabel, tokenRole } from './chatFold'

test('closed: the first token, then one fold item, then everything after the system turn', () => {
  expect(foldItems(6, [1, 4], false)).toEqual([
    { kind: 'token', index: 0 }, { kind: 'fold', count: 3 }, { kind: 'token', index: 4 }, { kind: 'token', index: 5 }])
})

test('open: the fold item stays in place and the folded tokens follow it', () => {
  expect(foldItems(4, [1, 3], true)).toEqual([
    { kind: 'token', index: 0 }, { kind: 'fold', count: 2 }, { kind: 'token', index: 1 },
    { kind: 'token', index: 2 }, { kind: 'token', index: 3 }])
})

test('without a system span every token is its own item', () => {
  expect(foldItems(3, undefined, false)).toEqual([0, 1, 2].map((index) => ({ kind: 'token', index })))
})

test('roles: special tokens by text, the user span as prompt, the rest as template; no spans → prompt', () => {
  const chat = { system: [1, 3] as [number, number], user: [5, 7] as [number, number] }
  expect(tokenRole(0, '<|im_start|>', chat)).toBe('special')
  expect(tokenRole(1, 'system', chat)).toBe('template')
  expect(tokenRole(5, 'What', chat)).toBe('prompt')
  expect(tokenRole(7, '<|im_end|>', chat)).toBe('special')
  expect(tokenRole(8, '\n', chat)).toBe('template')
  expect(tokenRole(0, '<|im_start|>', undefined)).toBe('prompt')
})

test('fold label', () => {
  expect(foldLabel(20, false)).toBe('system prompt · 20 tokens ▸')
  expect(foldLabel(20, true)).toBe('system prompt · 20 tokens ▾')
})
```

Append to `src/viz/TokenStream.test.tsx`:

```ts
test('a chat run folds the system turn after the first token; the chip opens and closes it', () => {
  const events = buildFixtureTrace({ chat: true })
  render(<TokenStream events={events} cursor={events.length - 1} />)
  const fold = screen.getByTestId('system-fold')
  expect(fold).toHaveTextContent('system prompt · 20 tokens ▸')
  expect(fold).toHaveAttribute('aria-expanded', 'false')
  expect(screen.getAllByTestId('prompt-token')).toHaveLength(37 - 20)
  expect(screen.getAllByTestId('prompt-token')[0]).toHaveTextContent('<|im_start|>')
  fireEvent.click(fold)
  expect(screen.getAllByTestId('prompt-token')).toHaveLength(37)
  expect(screen.getByTestId('system-fold')).toHaveTextContent('▾')
})

test('chat tokens carry their role; the generated <|im_end|> is marked as the end of the turn', () => {
  const events = buildFixtureTrace({ chat: true })
  render(<TokenStream events={events} cursor={events.length - 1} />)
  const prompt = screen.getAllByTestId('prompt-token')
  expect(prompt[0]).toHaveClass('token-special')
  expect(prompt.find((t) => t.textContent === 'user')).toHaveClass('token-template')
  expect(prompt.find((t) => t.textContent === 'What')).not.toHaveClass('token-template')
  const last = screen.getAllByTestId('generated-token').at(-1)!
  expect(last).toHaveClass('token-end')
  expect(last).toHaveAttribute('title', expect.stringContaining('end of turn'))
})

test('a chat run without a system turn has no fold chip', () => {
  const events = buildFixtureTrace({ chat: true })
  const t = events.find((e) => e.type === 'tokenize')!
  if (t.type === 'tokenize') t.chat = { user: [24, 31] }
  render(<TokenStream events={events} cursor={events.length - 1} />)
  expect(screen.queryByTestId('system-fold')).toBeNull()
})

test('a plain run looks as before: no fold chip, no template classes', () => {
  const events = buildFixtureTrace()
  render(<TokenStream events={events} cursor={events.length - 1} />)
  expect(screen.queryByTestId('system-fold')).toBeNull()
  expect(screen.getAllByTestId('prompt-token').every((t) => !t.className.includes('token-template'))).toBe(true)
})
```

(add `fireEvent` and `buildFixtureTrace` to the file's imports.)

Append to `src/viz/PipelineBand.test.tsx`:

```ts
test('the tokenizer card of a chat run says how many tokens are yours', () => {
  const events = buildFixtureTrace({ chat: true })
  render(<PipelineBand events={events} cursor={1} onStageClick={() => {}} />)
  expect(screen.getAllByTestId('stage-card')[0]).toHaveTextContent('37 tokens · 7 yours')
})
```

Append to `src/viz/DetailPanel.test.tsx`:

```ts
test('the tokenizer detail of a chat run says how many tokens the template added and folds the system turn', () => {
  const events = buildFixtureTrace({ chat: true, mode: 'real' })
  render(<DetailPanel events={events} cursor={1} mode="real" />)
  expect(screen.getByTestId('tokenizer-chat-note')).toHaveTextContent('30 of the 37 tokens were added by the template')
  expect(screen.getByTestId('tokenizer-chat-note').querySelector('[data-term="chat-template"]')).not.toBeNull()
  expect(screen.getByTestId('system-fold')).toHaveTextContent('system prompt · 20 tokens ▸')
})
```

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run src/viz/chatFold.test.ts src/viz/TokenStream.test.tsx src/viz/PipelineBand.test.tsx src/viz/DetailPanel.test.tsx`
Expected: FAIL (no module, no fold chip, old summary, no note).

- [ ] **Step 3: Implement `chatFold.ts`**

```ts
import type { ChatSpans } from '../trace/types'
import { isSpecialToken } from '../engine/chatTemplate'

// Folding a chat run's system turn: one item per shown token, plus one fold
// item where the system turn starts. Open, the folded tokens follow the item.
export type FoldItem = { kind: 'token'; index: number } | { kind: 'fold'; count: number }

export function foldItems(length: number, system: [number, number] | undefined, open: boolean): FoldItem[] {
  const all: FoldItem[] = Array.from({ length }, (_, index) => ({ kind: 'token', index }))
  if (!system || system[0] >= length) return all
  const fold: FoldItem = { kind: 'fold', count: system[1] - system[0] }
  return [...all.slice(0, system[0]), fold, ...all.slice(open ? system[0] : system[1])]
}

export type TokenRole = 'prompt' | 'template' | 'special'

// A prompt position's role in a chat run; every position of a plain run is 'prompt'.
export function tokenRole(index: number, text: string, chat: ChatSpans | undefined): TokenRole {
  if (!chat) return 'prompt'
  if (isSpecialToken(text)) return 'special'
  return index >= chat.user[0] && index < chat.user[1] ? 'prompt' : 'template'
}

export const foldLabel = (count: number, open: boolean): string =>
  `system prompt · ${count} tokens ${open ? '▾' : '▸'}`

export const FOLD_TITLE = 'The system turn the chat template adds before your message. See “Chat template” in the glossary.'
```

- [ ] **Step 4: Implement the strip**

`src/viz/TokenStream.tsx` becomes:

```tsx
import { useState } from 'react'
import type { TraceEvent } from '../trace/types'
import { chatLayout, distributionFor, visibleTokens } from './selectors'
import { TokenPopover } from './TokenPopover'
import { visibleToken } from './tokenText'
import { FOLD_TITLE, foldItems, foldLabel, tokenRole } from './chatFold'

const ROLE_CLASS = { prompt: '', template: ' token-template', special: ' token-special' } as const
const END_TOKEN = '<|im_end|>'

export function TokenStream({ events, cursor }: { events: TraceEvent[]; cursor: number }) {
  const { prompt, generated } = visibleTokens(events, cursor)
  const chat = chatLayout(events)
  const [open, setOpen] = useState(false)
  const [hoveredCycle, setHoveredCycle] = useState<number | null>(null)
  const dist = hoveredCycle !== null ? distributionFor(events, hoveredCycle) : undefined

  return (
    <div className="token-stream" aria-label="Token stream">
      {foldItems(prompt.length, chat?.system, open).map((item) => item.kind === 'fold' ? (
        <button key="fold" type="button" data-testid="system-fold" className="token token-fold"
          aria-expanded={open} title={FOLD_TITLE} onClick={() => setOpen((v) => !v)}>
          {foldLabel(item.count, open)}
        </button>
      ) : (
        <span key={`p${item.index}`} data-testid="prompt-token"
          className={`token token-prompt${ROLE_CLASS[tokenRole(item.index, prompt[item.index].text, chat)]}`}
          title={`id ${prompt[item.index].id}`}>
          {visibleToken(prompt[item.index])}
        </span>
      ))}
      {generated.map((t, i) => {
        const end = chat !== undefined && t.text === END_TOKEN
        return (
          <span key={`g${i}`} data-testid="generated-token"
            className={`token token-generated${end ? ' token-end' : ''}`}
            title={end ? `id ${t.id} · end of turn: the model stopped here` : `id ${t.id}`}
            onMouseEnter={() => setHoveredCycle(t.cycle)}
            onMouseLeave={() => setHoveredCycle(null)}>
            {visibleToken(t)}
            {hoveredCycle === t.cycle && dist && <TokenPopover softmax={dist.softmax} sample={dist.sample} />}
          </span>
        )
      })}
    </div>
  )
}
```

`src/index.css`, after `.token-generated { … }`:

```css
/* chat runs: the template's tokens dashed, special tokens in indigo, the end of the turn in vermillion */
.token-template { border-style: dashed; color: var(--ink-faint); background: transparent; }
.token-special { border-style: dashed; border-color: var(--ai-soft); color: var(--ai); background: transparent;
  font-family: var(--mono); font-size: .78em; }
.token-end { border-color: var(--shu); color: var(--shu); background: var(--shu-wash); font-family: var(--mono); font-size: .78em; }
.token-fold { border-style: dashed; color: var(--ink-soft); background: var(--washi); font: inherit; cursor: pointer; }
.token-fold:hover { border-color: var(--ai); }
.token-chip.token-template { border: 1px dashed var(--line-strong); background: transparent; }
```

- [ ] **Step 5: Implement the tokenizer detail and the summary**

`src/viz/details/TokenizerDetail.tsx`: add props `chat?: ChatSpans`; local `const [open, setOpen] = useState(false)`. After the existing first `<p>`, add:

```tsx
      {chat && (
        <p data-testid="tokenizer-chat-note">
          In Chat mode the prompt is wrapped in the model's <Term id="chat-template" section="what">chat template</Term>:{' '}
          {tokens.length - (chat.user[1] - chat.user[0])} of the {tokens.length} tokens were added by the template.
        </p>
      )}
```

and render the chips through `foldItems(tokens.length, chat?.system, open)`: a fold item becomes
`<button type="button" data-testid="system-fold" className="token-chip token-fold" aria-expanded={open} title={FOLD_TITLE} onClick={() => setOpen((v) => !v)}>{foldLabel(item.count, open)}</button>`;
a token item keeps today's chip markup, with `token-template` added to its class when `tokenRole(index, t.text, chat) !== 'prompt'`.

`src/viz/DetailPanel.tsx`: `<TokenizerDetail tokens={e.tokens} truncated={e.truncated} chat={e.chat} />`.

`src/viz/PipelineBand.tsx`, the tokenizer summary:

```ts
      return e ? (e.chat ? `${e.tokens.length} tokens · ${e.chat.user[1] - e.chat.user[0]} yours` : `${e.tokens.length} tokens`) : null
```

- [ ] **Step 6: Run the tests**

Run: `npx vitest run src/viz`
Expected: PASS (new and existing).

- [ ] **Step 7: Commit**

```bash
git add src/viz src/index.css
git commit -m "feat(chat): token strip and tokenizer detail fold the system turn and mark template, special and end tokens

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Fold the attention matrix; chat hint; spans into Layers and Compare

**Files:**
- Create: `src/viz/foldMatrix.ts`, `src/viz/foldMatrix.test.ts`
- Modify: `src/viz/AttentionHeatmap.tsx`, `src/viz/details/LayersDetail.tsx`, `src/viz/DetailPanel.tsx`, `src/viz/compare/CompareView.tsx`
- Test: `src/viz/AttentionHeatmap.test.tsx`

**Interfaces:**
- Consumes: `ChatSpans`, `chatLayout`, `buildFixtureTrace({ chat: true })`.
- Produces:
  - `interface FoldedMatrix { rows: number[]; cols: Array<{ from: number; to: number }>; values: number[][] }` — `values[i]` holds one entry per column whose `from <= rows[i]` (still ragged).
  - `foldMatrix(matrix: number[][], system: [number, number] | undefined): FoldedMatrix`
  - `AttentionHeatmap` props gain `chat?: ChatSpans` and `foldToggle?: boolean` (default `true`).
  - `LayersDetail` prop `chat?: ChatSpans`.

- [ ] **Step 1: Write the failing tests**

Create `src/viz/foldMatrix.test.ts`:

```ts
import { expect, test } from 'vitest'
import { foldMatrix } from './foldMatrix'

// 5 positions; row i spreads its weight evenly over 0..i
const M = Array.from({ length: 5 }, (_, i) => Array.from({ length: i + 1 }, () => 1 / (i + 1)))

test('without a span the matrix is unchanged, one column per position', () => {
  const f = foldMatrix(M, undefined)
  expect(f.rows).toEqual([0, 1, 2, 3, 4])
  expect(f.cols).toEqual([0, 1, 2, 3, 4].map((i) => ({ from: i, to: i + 1 })))
  expect(f.values).toEqual(M)
})

test('the span’s rows are hidden and its columns become one summed column after column 0', () => {
  const f = foldMatrix(M, [1, 3])
  expect(f.rows).toEqual([0, 3, 4])
  expect(f.cols).toEqual([{ from: 0, to: 1 }, { from: 1, to: 3 }, { from: 3, to: 4 }, { from: 4, to: 5 }])
  expect(f.values[0]).toEqual([1])                       // row 0 sees only column 0
  expect(f.values[1]).toEqual([0.25, 0.5, 0.25])         // row 3: col 0, cols 1–2 summed, col 3
  expect(f.values[2].reduce((a, b) => a + b, 0)).toBeCloseTo(1)
})

test('a matrix shorter than the span is not folded and never read past its rows', () => {
  const short = M.slice(0, 2)
  const f = foldMatrix(short, [1, 3])
  expect(f.rows).toEqual([0, 1])
  expect(f.values).toEqual(short)
})
```

Append to `src/viz/AttentionHeatmap.test.tsx`:

```ts
const chatTrace = buildFixtureTrace({ chat: true })
const chatAttn = chatTrace.filter((e) => e.type === 'attention').at(-1)
if (chatAttn?.type !== 'attention') throw new Error('chat fixture lacks attention')
const chatTokens = [...CHAT_PROMPT_TOKENS, ...CHAT_ANSWER_TOKENS]
const chat = { system: [1, 21] as [number, number], user: [24, 31] as [number, number] }

test('a chat run folds the system prompt into one column; the toggle opens the full matrix', () => {
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const n = chatAttn.heads[0].matrix.length
  expect(screen.getAllByTestId('row-label')).toHaveLength(n - 20)
  expect(screen.getAllByTestId('col-label').map((l) => l.textContent)).toContain('system prompt (20)')
  fireEvent.click(screen.getByTestId('attn-show-system'))
  expect(screen.getAllByTestId('row-label')).toHaveLength(n)
})

test('the folded column’s readout names the system prompt and its summed weight', () => {
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const cell = screen.getAllByTestId('attn-cell').find((c) => c.querySelector('title')?.textContent?.includes('→ system prompt (20)'))!
  fireEvent.mouseEnter(cell)
  expect(screen.getByTestId('attn-readout')).toHaveTextContent(/→ system prompt \(20\): \d+%/)
})

test('without spans there is no toggle and no fold; in compare there is no toggle either', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.queryByTestId('attn-show-system')).toBeNull()
  cleanup()
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} foldToggle={false} />)
  expect(screen.queryByTestId('attn-show-system')).toBeNull()
  expect(screen.getAllByTestId('col-label').map((l) => l.textContent)).toContain('system prompt (20)')
})

test('an induction head in a chat run explains that the template’s markers repeat', () => {
  const induction = [{ ...chatAttn.heads[0], label: 'induction' as const }]
  render(<AttentionHeatmap heads={induction} tokens={chatTokens} chat={chat} />)
  expect(screen.getByTestId('attn-hint')).toHaveTextContent('In Chat mode the template’s markers')
})
```

(add `buildFixtureTrace, CHAT_PROMPT_TOKENS, CHAT_ANSWER_TOKENS` to the fixture import.)

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run src/viz/foldMatrix.test.ts src/viz/AttentionHeatmap.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Implement `foldMatrix.ts`**

```ts
// A ragged causal attention matrix with a chat run's system turn folded: the
// span's rows hidden, its columns summed into one column right where it
// starts. Row 0 (the first <|im_start|>, the attention sink) stays.
export interface FoldedMatrix {
  rows: number[]                              // original row positions, in order
  cols: Array<{ from: number; to: number }>   // half-open position ranges, one per shown column
  values: number[][]                          // values[i][j] for every column with from <= rows[i]
}

export function foldMatrix(matrix: number[][], system: [number, number] | undefined): FoldedMatrix {
  const n = matrix.length
  const fold = system && system[1] <= n ? system : undefined
  const rows = Array.from({ length: n }, (_, i) => i).filter((i) => !fold || i < fold[0] || i >= fold[1])
  const cols: Array<{ from: number; to: number }> = []
  for (let i = 0; i < n; i++) {
    if (fold && i === fold[0]) { cols.push({ from: fold[0], to: fold[1] }); i = fold[1] - 1; continue }
    cols.push({ from: i, to: i + 1 })
  }
  const values = rows.map((r) => cols.filter((c) => c.from <= r).map((c) => {
    let sum = 0
    for (let k = c.from; k < Math.min(c.to, r + 1); k++) sum += matrix[r][k] ?? 0
    return sum
  }))
  return { rows, cols, values }
}
```

- [ ] **Step 4: Render the folded matrix in `AttentionHeatmap`**

In `src/viz/AttentionHeatmap.tsx`:
- Props: add `chat?: ChatSpans` and `foldToggle = true`.
- State: `const [showSystem, setShowSystem] = useState(false)`.
- `const folded = foldMatrix(head.matrix, showSystem && foldToggle ? undefined : chat?.system)`.
- Labels: `const rowLabel = (i: number) => label(folded.rows[i])`; `const colLabel = (j: number) => { const c = folded.cols[j]; return c.to - c.from > 1 ? \`system prompt (${c.to - c.from})\` : label(c.from) }`.
- Sizes from `folded.cols.length` (columns) and `folded.rows.length` (rows) instead of `n`.
- Column labels stay on the diagonal: column `j` is drawn above the first row that sees it, `const firstRow = folded.rows.findIndex((r) => r >= folded.cols[j].from)`, at `cy = DIAG_PAD + firstRow * CELL - 5`.
- Cells: iterate `folded.values[i]`, `x = LABEL_W + j * CELL`, `y = DIAG_PAD + i * CELL`; a folded column's cells get `strokeDasharray="2 2"` and `stroke="var(--ai)"` (class `attn-cell-fold`). Title and readout use `rowLabel(i)`, `colLabel(j)` and `folded.values[i][j]`.
- Hover state stores folded indices `{ r: i, c: j }`.
- Above the svg, only when `chat?.system && foldToggle`:

```tsx
        <label className="attn-fold-toggle">
          <input type="checkbox" data-testid="attn-show-system" checked={showSystem}
            onChange={(e) => { setShowSystem(e.target.checked); setHovered(null) }} />
          Show system prompt
        </label>
```

- Hint: `{HINTS[head.label]}{chat && head.label === 'induction' ? ' In Chat mode the template’s markers (<|im_start|>, ↵, <|im_end|>) repeat, and those repeats count too.' : ''}`.

`src/index.css`: `.attn-fold-toggle { display: inline-flex; gap: .3rem; align-items: center; font-size: .8rem; color: var(--ink-soft); margin: .2rem 0; }`.

- [ ] **Step 5: Pass the spans down**

- `src/viz/DetailPanel.tsx`, layers case: `chat={chatLayout(events)}` on `<LayersDetail … />` (import `chatLayout`).
- `src/viz/details/LayersDetail.tsx`: prop `chat?: ChatSpans`, passed as `<AttentionHeatmap heads={heads} tokens={tokens} focus={focus} chat={chat} />`.
- `src/viz/compare/CompareView.tsx`, `AttnSide`: `<AttentionHeatmap heads={[head]} tokens={runTokens(events)} chat={chatLayout(events)} foldToggle={false} />`.

- [ ] **Step 6: Run the tests**

Run: `npx vitest run src/viz`
Expected: PASS, including the existing heatmap tests (diagonal labels, hover readout, notes).

- [ ] **Step 7: Commit**

```bash
git add src/viz src/index.css
git commit -m "feat(chat): attention matrix folds the system prompt into one summed column with a Show system prompt toggle

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Fit the attention matrix to the card; hover without re-rendering cells

**Files:**
- Create: `src/viz/heatmapSizing.ts`, `src/viz/heatmapSizing.test.ts`
- Modify: `src/viz/AttentionHeatmap.tsx`
- Test: `src/viz/AttentionHeatmap.test.tsx`

**Interfaces:**
- Produces:
  - `MAX_CELL = 20`, `MIN_CELL = 8`, `LABEL_MIN = 14`
  - `cellSize(width: number | null, columns: number, chrome: number): number` — `chrome` = `LABEL_W + RIGHT_PAD`
  - `labelStep(cell: number): number`

- [ ] **Step 1: Write the failing tests**

Create `src/viz/heatmapSizing.test.ts`:

```ts
import { expect, test } from 'vitest'
import { cellSize, labelStep } from './heatmapSizing'

test('short runs keep 20 px cells', () => {
  expect(cellSize(928, 20, 136)).toBe(20)
  expect(cellSize(928, 39, 136)).toBe(20)
})

test('longer runs shrink to the card width, never below 8 px', () => {
  expect(cellSize(928, 57, 136)).toBe(13)    // floor(792 / 57)
  expect(cellSize(928, 137, 136)).toBe(8)
})

test('before the first measurement the cell is 20 px', () => {
  expect(cellSize(null, 137, 136)).toBe(20)
})

test('labels thin out below 14 px', () => {
  expect(labelStep(20)).toBe(1)
  expect(labelStep(14)).toBe(1)
  expect(labelStep(13)).toBe(2)
  expect(labelStep(8)).toBe(2)
  expect(labelStep(6)).toBe(3)
})
```

Append to `src/viz/AttentionHeatmap.test.tsx`:

```ts
test('a narrow card shrinks the cells and thins the labels; the hovered row and column keep theirs', () => {
  const resize = vi.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(400)
  render(<AttentionHeatmap heads={chatAttn.heads} tokens={chatTokens} chat={chat} />)
  const cells = screen.getAllByTestId('attn-cell')
  expect(Number(cells[0].getAttribute('width'))).toBeLessThan(19)
  const shown = screen.getAllByTestId('row-label').filter((l) => l.textContent !== '')
  expect(shown.length).toBeLessThan(screen.getAllByTestId('row-label').length)
  const unlabelled = screen.getAllByTestId('row-label').findIndex((l) => l.textContent === '')
  const cell = cells.find((c) => c.getAttribute('data-row') === String(unlabelled))!
  fireEvent.mouseEnter(cell)
  expect(screen.getAllByTestId('row-label')[unlabelled].textContent).not.toBe('')
  resize.mockRestore()
})

test('hovering draws one outline on top and leaves the cells untouched', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.queryByTestId('attn-hover')).toBeNull()
  const cell = screen.getAllByTestId('attn-cell')[1]
  fireEvent.mouseEnter(cell)
  expect(screen.getByTestId('attn-hover')).toHaveAttribute('x', cell.getAttribute('x'))
  expect(screen.getAllByTestId('attn-cell').every((c) => !c.hasAttribute('stroke-width'))).toBe(true)
})
```

(add `vi` to the vitest import. The component reads the card width from `clientWidth` of its wrapper on mount and on `ResizeObserver` callbacks; jsdom has no `ResizeObserver`, so the component guards it and reads `clientWidth` once in a layout effect.)

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run src/viz/heatmapSizing.test.ts src/viz/AttentionHeatmap.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Implement the sizing**

`src/viz/heatmapSizing.ts`:

```ts
// The attention matrix fits its card: 20 px cells while they fit, smaller
// down to 8 px, after that the matrix scrolls sideways as before.
export const MAX_CELL = 20
export const MIN_CELL = 8
// below this, a label on every row and column would overlap its neighbours
export const LABEL_MIN = 14

export function cellSize(width: number | null, columns: number, chrome: number): number {
  if (width === null || columns === 0) return MAX_CELL
  return Math.max(MIN_CELL, Math.min(MAX_CELL, Math.floor((width - chrome) / columns)))
}

export const labelStep = (cell: number): number => (cell >= LABEL_MIN ? 1 : Math.ceil(LABEL_MIN / cell))
```

- [ ] **Step 4: Use it in `AttentionHeatmap`**

- Replace the module constant `CELL` with the computed `cell`. Wrapper `<div ref={wrapRef} …>`; width state:

```tsx
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState<number | null>(null)
  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    setWidth(el.clientWidth || null)
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth || null))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const cell = cellSize(width, folded.cols.length, LABEL_W + RIGHT_PAD)
  const step = labelStep(cell)
```

- Row label `i` text: `i % step === 0 || hovered?.r === i ? rowLabel(i) : ''`; column label `j`: `j % step === 0 || hovered?.c === j ? colLabel(j) : ''`.
- Cells move into a memoised child so hover changes do not re-render them:

```tsx
const Cells = memo(function Cells({ folded, cell, onHover, rowLabel, colLabel }: {
  folded: FoldedMatrix; cell: number; onHover: (h: { r: number; c: number } | null) => void
  rowLabel: (i: number) => string; colLabel: (j: number) => string
}) {
  return <>{folded.values.map((row, r) => row.map((w, c) => {
    const fold = folded.cols[c].to - folded.cols[c].from > 1
    return (
      <rect key={`${r}-${c}`} data-testid="attn-cell" data-row={r} data-col={c}
        x={LABEL_W + c * cell} y={DIAG_PAD + r * cell} width={cell - 1} height={cell - 1}
        fill={`hsl(211 ${Math.round(30 + 25 * Math.min(1, w))}% ${Math.round(94 - 70 * Math.min(1, w))}%)`}
        className={fold ? 'attn-cell-fold' : undefined}
        onMouseEnter={() => onHover({ r, c })} onMouseLeave={() => onHover(null)}>
        <title>{`${rowLabel(r)} → ${colLabel(c)}: ${Math.round(w * 100)}%`}</title>
      </rect>
    )
  }))}</>
})
```

  `rowLabel` and `colLabel` passed here are the unthinned labels, wrapped in `useCallback` keyed on `folded` and `tokens`; `folded` is wrapped in `useMemo` keyed on `head.matrix`, `chat?.system`, `showSystem`, `foldToggle`; `onHover` is the state setter `setHovered`.
- After the cells, the outline:

```tsx
        {hovered && (
          <rect data-testid="attn-hover" className="attn-hover" pointerEvents="none"
            x={LABEL_W + hovered.c * cell} y={DIAG_PAD + hovered.r * cell} width={cell - 1} height={cell - 1} />
        )}
```

- `src/index.css`: `.attn-hover { fill: none; stroke: #d64; stroke-width: 2; }` and `.attn-cell-fold { stroke: var(--ai); stroke-dasharray: 2 2; }`.

- [ ] **Step 5: Run the tests**

Run: `npx vitest run src/viz`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/viz src/index.css
git commit -m "feat(attention): cells fit the card from 20 down to 8 px, labels thin out, hover draws one outline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Fold the system rows in Edit sizes

**Files:**
- Modify: `src/viz/EditSizeCard.tsx`, `src/viz/details/LayersDetail.tsx`, `src/index.css`
- Test: `src/viz/EditSizeCard.test.tsx`

**Interfaces:**
- Consumes: `foldItems`, `foldLabel`, `FOLD_TITLE` (Task 5), `chat` prop already on `LayersDetail` (Task 6).
- Produces: `EditSizeCard` prop `chat?: ChatSpans`; row button `data-testid="edit-fold"`.

- [ ] **Step 1: Write the failing tests**

Append to `src/viz/EditSizeCard.test.tsx`:

```ts
const chatRows = (n: number) => ({
  tokens: Array.from({ length: n }, (_, i) => ({ id: i, text: `t${i}` })),
  attn: Array.from({ length: n }, () => [0.1, 0.2, 0.3]),
  mlp: Array.from({ length: n }, () => [0.4, 0.5, 0.6]),
})
const chat = { system: [1, 21] as [number, number], user: [24, 31] as [number, number] }

test('a chat run folds the system rows into one row that opens on click', () => {
  render(<EditSizeCard rows={chatRows(45)} chat={chat} note="unused" />)
  const fold = screen.getByTestId('edit-fold')
  expect(fold).toHaveTextContent('system prompt · 20 tokens ▸')
  expect(screen.getAllByTestId('edit-row-label')).toHaveLength(25)
  fireEvent.click(fold)
  expect(screen.getAllByTestId('edit-row-label')).toHaveLength(45)
})

test('the 64-row cap applies after folding', () => {
  render(<EditSizeCard rows={chatRows(80)} chat={chat} note="unused" />)
  expect(screen.getAllByTestId('edit-row-label')).toHaveLength(60)   // 80 − 20 fit under the cap
  expect(screen.queryByTestId('edit-rows-note')).toBeNull()
  cleanup()
  render(<EditSizeCard rows={chatRows(100)} chat={chat} note="unused" />)
  expect(screen.getAllByTestId('edit-row-label')).toHaveLength(64)
  expect(screen.queryByTestId('edit-fold')).toBeNull()               // the fold row lies before the cut
  expect(screen.getByTestId('edit-rows-note')).toHaveTextContent('showing the last 64 of 100 tokens')
})

test('the fold row is not selectable; the bars keep showing the selected token', () => {
  render(<EditSizeCard rows={chatRows(45)} chat={chat} note="unused" />)
  fireEvent.click(screen.getByTestId('edit-fold'))
  fireEvent.click(screen.getByTestId('edit-fold'))
  expect(screen.getByTestId('edit-bars-label')).toHaveTextContent('t44')
})
```

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run src/viz/EditSizeCard.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `src/viz/EditSizeCard.tsx`:
- Remove `lastRows`. Keep `MAX_ROWS = 64`. Add:

```ts
// The last max token items; a fold item before the cut is dropped with it.
function capItems(items: FoldItem[], max: number): FoldItem[] {
  let tokens = 0
  for (let i = items.length - 1; i >= 0; i--)
    if (items[i].kind === 'token' && ++tokens === max) return items.slice(i)
  return items
}
```

- In `EditSizeCard`: props add `chat?: ChatSpans`; state `const [open, setOpen] = useState(false)`. With `all` present:

```ts
  const n = all ? all.tokens.length : 0
  const items = all ? capItems(foldItems(n, chat?.system, open), MAX_ROWS) : []
  const shown = items.filter((it) => it.kind === 'token').length
  const last = n - 1
  const sel = picked !== null && picked <= last ? picked : last
```

  `picked`/`sel` are positions in `all` (no longer indices into a cut copy). Pass `items`, `all`, `open` and `onToggle={() => setOpen((v) => !v)}` to `EditHeatmap`; `EditBars` gets `rows={all}`. The note shows when `shown < n`: `showing the last {shown} of {n} tokens`.
- `EditHeatmap` iterates `items`: a token item renders today's row for position `item.index` (`data-row={item.index}`, `aria-pressed={item.index === selected}`, `onSelect(item.index)`); a fold item renders

```tsx
        <div key="fold" className="mlp-heatmap-row">
          <button type="button" data-testid="edit-fold" className="mlp-tok mlp-fold" aria-expanded={open}
            title={FOLD_TITLE} onClick={onToggle}>{foldLabel(item.count, open)}</button>
          <div className="mlp-cells mlp-cells-empty" />
        </div>
```

- `src/index.css`: `.mlp-fold { border-style: dashed; color: var(--ink-soft); }` and `.mlp-cells-empty { min-height: 1px; }`.
- `src/viz/details/LayersDetail.tsx`: `<EditSizeCard … chat={chat} />`.

- [ ] **Step 4: Run the tests**

Run: `npx vitest run src/viz/EditSizeCard.test.tsx src/viz/DetailPanel.test.tsx`
Expected: PASS, including the existing "last 64 tokens" and selection tests.

- [ ] **Step 5: Commit**

```bash
git add src/viz src/index.css
git commit -m "feat(chat): Edit sizes folds the system rows; the 64-row cap applies after folding

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Mode switch, format switch, system prompt and the chat example

**Files:**
- Create: `src/app/chatExample.ts`
- Modify: `src/app/PromptBar.tsx`, `src/App.tsx`, `src/index.css`
- Test: `src/app/PromptBar.test.tsx`, `src/App.modeGuard.test.tsx`, `src/App.test.tsx`

**Interfaces:**
- Consumes: `PromptFormat`, `TEXT_FORMAT`, `DEFAULT_SYSTEM_PROMPT`, `systemOf` (Task 1); `engine.run(…, format)` (Task 3); `<Term id="chat-template">` (Task 2).
- Produces:
  - `PromptExample.format?: 'chat' | 'text'`
  - `PromptBarProps.onGenerate(prompt: string, params: GenParams, format: PromptFormat): void`
  - Test ids: `mode-switch`, `mode-sim`, `mode-real` (buttons with `aria-pressed`), `format-switch`, `format-chat`, `format-text`, `system-input`, `system-reset`, `example-chip-tag`.
  - `CHAT_EXAMPLE: PromptExample` in `src/app/chatExample.ts`.
  - `App.handleGenerate(prompt, params, format = TEXT_FORMAT)`: a chat request with no real engine loads it first and runs only if loading succeeded.

- [ ] **Step 1: Write the failing tests**

In `src/app/PromptBar.test.tsx`:
- Replace the test `'mode toggle reports changes'` with:

```ts
test('the mode switch reports changes and shows the current mode', () => {
  const onModeChange = vi.fn()
  render(<PromptBar mode="sim" onModeChange={onModeChange} onGenerate={noop} busy={false} />)
  expect(screen.getByTestId('mode-sim')).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByTestId('mode-real')).toHaveTextContent('Real model · 250 MB')
  fireEvent.click(screen.getByTestId('mode-real'))
  expect(onModeChange).toHaveBeenCalledWith('real')
})
```

- Change the two existing `toHaveBeenCalledWith('…', { … })` expectations (sim mode) to add a third argument `{ kind: 'text' }`.
- Append:

```ts
test('the format switch and the system line exist only with the real model', () => {
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={noop} busy={false} />)
  expect(screen.queryByTestId('format-switch')).toBeNull()
  expect(screen.queryByTestId('system-input')).toBeNull()
  cleanup()
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={noop} busy={false} />)
  expect(screen.getByTestId('format-chat')).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByTestId('system-input')).toHaveValue('You are a helpful AI assistant named SmolLM, trained by Hugging Face')
  expect(screen.getByTestId('prompt-input')).toHaveAttribute('placeholder', 'Ask a question…')
  fireEvent.click(screen.getByTestId('format-text'))
  expect(screen.queryByTestId('system-input')).toBeNull()
  expect(screen.getByTestId('prompt-input')).toHaveAttribute('placeholder', 'Type a prompt…')
})

test('Chat sends the system prompt; empty means none; Reset restores the default', () => {
  const onGenerate = vi.fn()
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'Hi' } })
  expect(screen.queryByTestId('system-reset')).toBeNull()
  fireEvent.change(screen.getByTestId('system-input'), { target: { value: '  ' } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenLastCalledWith('Hi', expect.anything(), { kind: 'chat', system: null })
  fireEvent.click(screen.getByTestId('system-reset'))
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenLastCalledWith('Hi', expect.anything(),
    { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' })
})

test('Chat selected, then Simulated: the run is plain text', () => {
  const onGenerate = vi.fn()
  const { rerender } = render(<PromptBar mode="real" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  rerender(<PromptBar mode="sim" onModeChange={noop} onGenerate={onGenerate} busy={false} />)
  fireEvent.change(screen.getByTestId('prompt-input'), { target: { value: 'Hi' } })
  fireEvent.click(screen.getByTestId('btn-generate'))
  expect(onGenerate).toHaveBeenLastCalledWith('Hi', expect.anything(), { kind: 'text' })
})

test('a text example sets Continue text; the chat example sets Chat with the default system prompt', () => {
  const onGenerate = vi.fn()
  const examples = [{ id: 't', label: 'Text', prompt: 'one two', hint: 'h' }, CHAT_EXAMPLE]
  render(<PromptBar mode="real" onModeChange={noop} onGenerate={onGenerate} busy={false} examples={examples} />)
  fireEvent.click(screen.getAllByTestId('example-chip')[0])
  expect(onGenerate).toHaveBeenLastCalledWith('one two', expect.anything(), { kind: 'text' })
  expect(screen.getByTestId('format-text')).toHaveAttribute('aria-pressed', 'true')
  fireEvent.click(screen.getAllByTestId('example-chip')[1])
  expect(screen.getByTestId('format-chat')).toHaveAttribute('aria-pressed', 'true')
  expect(onGenerate).toHaveBeenLastCalledWith('What is the capital of France?', expect.anything(),
    { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' })
  expect(screen.getAllByTestId('example-chip')[1].querySelector('[data-testid="example-chip-tag"]')).toHaveTextContent('Chat')
})

test('the chat example in simulated mode still asks for a chat run (the app loads the model first)', () => {
  const onGenerate = vi.fn()
  render(<PromptBar mode="sim" onModeChange={noop} onGenerate={onGenerate} busy={false} examples={[CHAT_EXAMPLE]} />)
  fireEvent.click(screen.getByTestId('example-chip'))
  expect(onGenerate).toHaveBeenLastCalledWith('What is the capital of France?', expect.anything(),
    expect.objectContaining({ kind: 'chat' }))
})
```

(import `CHAT_EXAMPLE` from `./chatExample`.)

In `src/App.modeGuard.test.tsx`: the three tests click `mode-toggle`. The rapid-toggle test clicks `mode-real`, `mode-sim`, `mode-real`, `mode-sim`, `mode-real` in turn; the failed-prepare test clicks `mode-real` and then waits for `expect(screen.getByTestId('mode-real')).toHaveAttribute('aria-pressed', 'false')`; the loading test clicks `mode-real`. Then append, after replacing the fake engine's `run` with one that records its arguments (`runs.push(args)` and returns `{ abort() {}, done: Promise.resolve() }`, `let runs: unknown[][] = []` reset in `beforeEach`):

```ts
test('the chat example with the real model off loads it first, then runs as chat', async () => {
  render(<App />)
  fireEvent.click(screen.getAllByTestId('example-chip').at(-1)!)
  expect(constructCount).toBe(1)
  expect(runs).toHaveLength(0)
  deferreds[0].resolve()
  await waitFor(() => expect(runs).toHaveLength(1))
  expect(runs[0][0]).toBe('What is the capital of France?')
  expect(runs[0][3]).toMatchObject({ kind: 'chat' })
})

test('if loading fails, the chat example runs nothing', async () => {
  render(<App />)
  fireEvent.click(screen.getAllByTestId('example-chip').at(-1)!)
  deferreds[0].reject(new Error('download failed'))
  await waitFor(() => expect(screen.getByTestId('model-error')).toBeInTheDocument())
  expect(runs).toHaveLength(0)
})
```

In `src/App.test.tsx`, line ~102 clicks `mode-toggle`: click `mode-real` instead.

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run src/app/PromptBar.test.tsx src/App.modeGuard.test.tsx src/App.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Implement the chat example**

`src/app/chatExample.ts`:

```ts
import type { PromptExample } from './PromptBar'

// The one example that runs in Chat mode; it needs the real model.
export const CHAT_EXAMPLE: PromptExample = {
  id: 'chat-question',
  label: 'Chat: a question',
  prompt: 'What is the capital of France?',
  format: 'chat',
  hint: 'As Continue text this question usually ends at once: the model\'s first token is <|im_end|> in about 9 of 10 runs. In Chat mode the template\'s tokens tell it a user asked and the assistant should reply. It answers, and often ends its turn itself with <|im_end|>; with T = 0 it always does.',
}
```

- [ ] **Step 4: Implement `PromptBar`**

- `PromptExample` gains `format?: 'chat' | 'text'`; `onGenerate(prompt: string, params: GenParams, format: PromptFormat): void`.
- State: `const [chatFmt, setChatFmt] = useState(true)`, `const [system, setSystem] = useState(DEFAULT_SYSTEM_PROMPT)`.
- `const chatOn = mode === 'real' && chatFmt`; `const format: PromptFormat = chatOn ? { kind: 'chat', system: systemOf(system) } : TEXT_FORMAT`.
- The prompt row, in this order: mode switch, format switch (only `mode === 'real'`), the input (placeholder `chatOn ? 'Ask a question…' : 'Type a prompt…'`), Generate, Stop:

```tsx
        <div className="seg" role="group" aria-label="Model" data-testid="mode-switch">
          <button type="button" data-testid="mode-sim" aria-pressed={mode === 'sim'}
            onClick={() => { if (mode !== 'sim') onModeChange('sim') }}>Simulated</button>
          <button type="button" data-testid="mode-real" aria-pressed={mode === 'real'} title={REAL_TITLE}
            onClick={() => { if (mode !== 'real') onModeChange('real') }}>Real model <span className="mode-note">· 250 MB</span></button>
        </div>
        {mode === 'real' && (
          <div className="seg" role="group" aria-label="Prompt format" data-testid="format-switch">
            <button type="button" data-testid="format-chat" aria-pressed={chatFmt} onClick={() => setChatFmt(true)}>Chat</button>
            <button type="button" data-testid="format-text" aria-pressed={!chatFmt} onClick={() => setChatFmt(false)}>Continue text</button>
            <Term id="chat-template" variant="icon" label="About the chat template" testId="format-info">ⓘ</Term>
          </div>
        )}
```

  `REAL_TITLE` is today's checkbox `title` text, moved as is.
- After the prompt row, only when `chatOn`:

```tsx
      {chatOn && (
        <div className="system-row">
          <label htmlFor="system-input">System prompt</label>
          <input id="system-input" data-testid="system-input" value={system} onChange={(e) => setSystem(e.target.value)} />
          {system !== DEFAULT_SYSTEM_PROMPT && (
            <button type="button" data-testid="system-reset" className="link-button"
              onClick={() => setSystem(DEFAULT_SYSTEM_PROMPT)}>Reset to default</button>
          )}
        </div>
      )}
```

- Remove the `Real model` checkbox label from the config row (the `status` slot stays there until Task 10).
- Generate: `onGenerate(prompt, { temperature, topK, maxNewTokens }, format)`.
- Example click:

```tsx
              onClick={() => {
                setPrompt(ex.prompt)
                const params = { temperature, topK, maxNewTokens }
                if (ex.format === 'chat') {
                  setChatFmt(true); setSystem(DEFAULT_SYSTEM_PROMPT)
                  onGenerate(ex.prompt, params, { kind: 'chat', system: DEFAULT_SYSTEM_PROMPT })
                } else {
                  setChatFmt(false)
                  onGenerate(ex.prompt, params, TEXT_FORMAT)
                }
              }}>
              {ex.format === 'chat' && <span data-testid="example-chip-tag" className="example-chip-tag">Chat</span>}
              {ex.label}
```

- `src/index.css`:

```css
.seg { display: inline-flex; align-items: center; border: 1px solid var(--ai-soft); border-radius: 6px; overflow: hidden; }
.seg > button { border: 0; border-radius: 0; background: var(--card); color: var(--ai); padding: .35rem .7rem; font-size: .85rem; }
.seg > button[aria-pressed="true"] { background: var(--ai); color: #fff; }
.seg > button[aria-pressed="true"] .mode-note { color: var(--ai-wash-strong); }
.seg .gl-term-wrap { padding: 0 .4rem; }
.system-row { display: flex; gap: .5rem; align-items: center; font-size: .85rem; color: var(--ink-soft); }
.system-row input { flex: 1; min-width: 12rem; padding: .3rem .5rem; }
.link-button { border: 0; background: none; color: var(--ai); text-decoration: underline; padding: 0; font-size: .85rem; cursor: pointer; }
.example-chip-tag { font-size: .68rem; color: var(--ai); border: 1px solid var(--ai-soft); border-radius: 3px;
  padding: 0 .2rem; margin-right: .3rem; background: var(--card); }
```

- [ ] **Step 5: Implement the App side**

In `src/App.tsx`:
- `examples={[...CURATED_EXAMPLES, CHAT_EXAMPLE]}`.
- `handleModeChange` returns `Promise<boolean>`: `true` when the real engine is ready at the end, `false` otherwise (each branch returns accordingly).
- `handleGenerate`:

```ts
  const handleGenerate = async (prompt: string, params: GenParams, format: PromptFormat = TEXT_FORMAT) => {
    // the chat example can be clicked while simulated: load the real model first
    if (format.kind === 'chat' && !realEngineRef.current && !(await handleModeChange('real'))) return
    runRef.current?.abort()
    ...
    const engine: PipelineEngine =
      (format.kind === 'chat' || mode === 'real') && realEngineRef.current
        ? realEngineRef.current
        : new SimulatedEngine(tokenizerRef.current)
    try {
      const handle = engine.run(prompt, params, (e) => { … }, format)
```

- [ ] **Step 6: Run the tests**

Run: `npx vitest run src/app src/App.test.tsx src/App.modeGuard.test.tsx`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/app src/App.tsx src/App.test.tsx src/App.modeGuard.test.tsx src/index.css
git commit -m "feat(chat): Simulated | Real model switch, Chat | Continue text switch, editable system prompt, chat example

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Make the real model hard to miss

**Files:**
- Create: `src/app/realModel.ts`, `src/app/realModel.test.ts`, `src/app/LoadRealButton.tsx`, `src/app/ModelSlot.tsx`, `src/app/ModelSlot.test.tsx`, `src/app/ModeBadge.tsx`, `src/app/ModeBadge.test.tsx`
- Delete: `src/app/ModelStatus.tsx`, `src/app/ModelStatus.test.tsx` (their cases move to `ModelSlot.test.tsx` and `ModeBadge.test.tsx`)
- Modify: `src/App.tsx`, `src/app/PromptBar.tsx` (drop `status`), `src/viz/details/LayersDetail.tsx`, `src/viz/EditSizeCard.tsx`, `src/viz/LayerVotesCard.tsx`, `src/viz/NeuronCard.tsx`, `src/viz/MlpStepsCard.tsx`, `src/viz/AttentionHeatmap.tsx`, `src/index.css`
- Test: `src/App.modeGuard.test.tsx`, `src/viz/details` via `src/viz/DetailPanel.test.tsx`, `src/viz/AttentionHeatmap.test.tsx`, `src/app/PromptBar.test.tsx`

**Interfaces:**
- Produces (`src/app/realModel.ts`):
  - `useRealModel`: zustand store `{ request: () => void; loading: boolean }`
  - `readFlag(key: PrefKey): boolean`, `writeFlag(key: PrefKey, on: boolean): void`, `type PrefKey = 'tsumugi.simBannerClosed' | 'tsumugi.realModelUsed'`
  - `MODEL_FILE_URL: string` and `realModelCached(store?: CacheStorage): Promise<boolean>`
- Produces components: `LoadRealButton({ variant?: 'button' | 'link' })`, `ModelSlot({ mode, loading, fromCache, progress, error, bannerClosed, onLoad, onClose, onFallback })`, `ModeBadge({ mode, loading, device, attentions })`.
- The four MLP cards gain `noteAction?: ReactNode`, rendered at the end of their note paragraph.

- [ ] **Step 1: Write the failing tests**

`src/app/realModel.test.ts`:

```ts
import { afterEach, expect, test, vi } from 'vitest'
import { MODEL_FILE_URL, readFlag, realModelCached, writeFlag } from './realModel'

afterEach(() => { localStorage.clear(); vi.restoreAllMocks() })

test('flags round-trip through localStorage', () => {
  expect(readFlag('tsumugi.realModelUsed')).toBe(false)
  writeFlag('tsumugi.realModelUsed', true)
  expect(readFlag('tsumugi.realModelUsed')).toBe(true)
  writeFlag('tsumugi.realModelUsed', false)
  expect(readFlag('tsumugi.realModelUsed')).toBe(false)
})

test('storage that throws reads as unset and never throws', () => {
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
  expect(readFlag('tsumugi.simBannerClosed')).toBe(false)
  expect(() => writeFlag('tsumugi.simBannerClosed', true)).not.toThrow()
})

test('the model counts as cached only when its file is in the transformers cache', async () => {
  const fake = (has: boolean, file: boolean) => ({
    has: async () => has,
    open: async () => ({ match: async (url: string) => (file && url === MODEL_FILE_URL ? new Response('x') : undefined) }),
  }) as unknown as CacheStorage
  expect(MODEL_FILE_URL).toBe('https://huggingface.co/saigyo-hoshi/smollm2-135m-attn-onnx/resolve/v6/onnx/model_quantized.onnx')
  expect(await realModelCached(fake(true, true))).toBe(true)
  expect(await realModelCached(fake(true, false))).toBe(false)
  expect(await realModelCached(fake(false, false))).toBe(false)
  expect(await realModelCached(undefined)).toBe(false)
})
```

`src/app/ModelSlot.test.tsx`:

```tsx
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { ModelSlot } from './ModelSlot'

afterEach(cleanup)
const base = { mode: 'sim' as const, loading: false, fromCache: false, progress: null, error: null,
  bannerClosed: false, onLoad: () => {}, onClose: () => {}, onFallback: () => {} }

test('simulated: the banner explains the simulation and offers to load the real model', () => {
  const onLoad = vi.fn(), onClose = vi.fn()
  render(<ModelSlot {...base} onLoad={onLoad} onClose={onClose} />)
  expect(screen.getByTestId('sim-banner')).toHaveTextContent('You are watching a simulation.')
  expect(screen.getByTestId('sim-banner')).toHaveTextContent('250 MB, downloaded once, runs entirely in your browser.')
  fireEvent.click(screen.getByTestId('banner-load'))
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect(onLoad).toHaveBeenCalledOnce()
  expect(onClose).toHaveBeenCalledOnce()
})

test('closed banner or real mode: the slot is empty', () => {
  const { container } = render(<ModelSlot {...base} bannerClosed />)
  expect(container).toBeEmptyDOMElement()
  cleanup()
  const real = render(<ModelSlot {...base} mode="real" />)
  expect(real.container).toBeEmptyDOMElement()
})

test('loading shows the file and percentage even after the banner was closed', () => {
  render(<ModelSlot {...base} mode="real" loading bannerClosed progress={{ file: 'onnx/model_quantized.onnx', loaded: 42, total: 100 }} />)
  expect(screen.getByTestId('model-progress')).toHaveTextContent('Loading the real model: onnx/model_quantized.onnx · 42 %')
  expect(screen.getByTestId('model-progress-bar')).toHaveAttribute('value', '42')
})

test('loading from the cache says so', () => {
  render(<ModelSlot {...base} mode="real" loading fromCache />)
  expect(screen.getByTestId('model-progress')).toHaveTextContent('Loading the real model from the browser’s cache…')
})

test('an error shows there with the fallback button', () => {
  const onFallback = vi.fn()
  render(<ModelSlot {...base} error="download failed" onFallback={onFallback} />)
  expect(screen.getByTestId('model-error')).toHaveTextContent('Real model unavailable: download failed')
  fireEvent.click(screen.getByTestId('btn-fallback'))
  expect(onFallback).toHaveBeenCalledOnce()
})
```

`src/app/ModeBadge.test.tsx`:

```tsx
import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { ModeBadge } from './ModeBadge'

afterEach(cleanup)

test('badge texts for simulated, loading and ready', () => {
  render(<ModeBadge mode="sim" loading={false} device={null} attentions={false} />)
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Simulated')
  cleanup()
  render(<ModeBadge mode="real" loading device={null} attentions={false} />)
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Loading…')
  cleanup()
  render(<ModeBadge mode="real" loading={false} device="webgpu" attentions />)
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Real model · webgpu · attn')
})
```

Append to `src/viz/DetailPanel.test.tsx`:

```ts
test('in a simulated run the real-model-only notes offer to load the real model', () => {
  const request = vi.fn()
  useRealModel.setState({ request, loading: false })
  const events = buildFixtureTrace()
  const layer = events.findIndex((e) => e.type === 'layer')
  render(<DetailPanel events={events} cursor={layer} mode="sim" />)
  const buttons = screen.getAllByTestId('load-real')
  expect(buttons.length).toBeGreaterThanOrEqual(4)
  fireEvent.click(buttons[0])
  expect(request).toHaveBeenCalledOnce()
  useRealModel.setState({ request, loading: true })
  expect(screen.getAllByTestId('load-real').every((b) => (b as HTMLButtonElement).disabled)).toBe(true)
})
```

(import `useRealModel` from `../app/realModel`, `vi`, `fireEvent`.)

Append to `src/viz/AttentionHeatmap.test.tsx`:

```ts
test('the simulated note says the real model would show computed weights, with a link to load it', () => {
  render(<AttentionHeatmap heads={attn.heads} tokens={tokens} />)
  expect(screen.getByTestId('attn-note')).toHaveTextContent('Illustrative pattern (simulated): the real model would show the weights it computed.')
  expect(screen.getByTestId('attn-note').querySelector('[data-testid="load-real"]')).not.toBeNull()
})
```

(Update the existing test `'unscored heads keep the illustrative note'` if it matches the old text.)

In `src/App.modeGuard.test.tsx`, append (with `vi.mock('./app/realModel', …)` keeping the real module but making `realModelCached` controllable: `let cached = false` reset in `beforeEach`, and the mock `realModelCached: async () => cached`; also `localStorage.clear()` in `beforeEach`):

```ts
test('first visit: the banner shows; closing it is remembered', async () => {
  const { unmount } = render(<App />)
  expect(screen.getByTestId('sim-banner')).toBeInTheDocument()
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Simulated')
  fireEvent.click(screen.getByRole('button', { name: 'Close' }))
  expect(screen.queryByTestId('sim-banner')).toBeNull()
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Simulated')
  unmount()
  render(<App />)
  expect(screen.queryByTestId('sim-banner')).toBeNull()
})

test('a returning user with the model cached starts in real mode without a click', async () => {
  localStorage.setItem('tsumugi.realModelUsed', '1')
  cached = true
  render(<App />)
  await waitFor(() => expect(constructCount).toBe(1))
  expect(screen.getByTestId('model-progress')).toHaveTextContent('from the browser’s cache')
  deferreds[0].resolve()
  await waitFor(() => expect(screen.getByTestId('mode-badge')).toHaveTextContent('Real model · wasm'))
})

test('a returning user whose cache was cleared starts simulated with the banner and downloads nothing', async () => {
  localStorage.setItem('tsumugi.realModelUsed', '1')
  localStorage.setItem('tsumugi.simBannerClosed', '1')
  cached = false
  render(<App />)
  await waitFor(() => expect(screen.getByTestId('sim-banner')).toBeInTheDocument())
  expect(constructCount).toBe(0)
})

test('the banner’s button loads the model; ready sets the remembered flag', async () => {
  render(<App />)
  fireEvent.click(screen.getByTestId('banner-load'))
  expect(constructCount).toBe(1)
  deferreds[0].resolve()
  await waitFor(() => expect(localStorage.getItem('tsumugi.realModelUsed')).toBe('1'))
})
```

In `src/app/PromptBar.test.tsx`, delete the test `'status slot renders inside the config row'`.

- [ ] **Step 2: Run them to see them fail**

Run: `npx vitest run src/app src/App.modeGuard.test.tsx src/viz/DetailPanel.test.tsx src/viz/AttentionHeatmap.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Implement `realModel.ts`**

```ts
import { create } from 'zustand'
import { ATTN_MODEL_ID, ATTN_MODEL_REVISION } from '../engine/tokenizer'

// Any card can ask the app to load the real model; the app installs the
// handler and keeps loading up to date.
interface RealModelState { request: () => void; loading: boolean }
export const useRealModel = create<RealModelState>(() => ({ request: () => {}, loading: false }))

export type PrefKey = 'tsumugi.simBannerClosed' | 'tsumugi.realModelUsed'

// Per-browser conveniences: without storage the app behaves as on a first visit.
export function readFlag(key: PrefKey): boolean {
  try { return localStorage.getItem(key) === '1' } catch { return false }
}
export function writeFlag(key: PrefKey, on: boolean): void {
  try { if (on) localStorage.setItem(key, '1'); else localStorage.removeItem(key) } catch { /* session only */ }
}

// transformers.js caches each file under its download URL in the Cache
// Storage named by env.cacheKey ('transformers-cache').
export const MODEL_FILE_URL =
  `https://huggingface.co/${ATTN_MODEL_ID}/resolve/${ATTN_MODEL_REVISION}/onnx/model_quantized.onnx`

export async function realModelCached(
  store: CacheStorage | undefined = typeof caches === 'undefined' ? undefined : caches,
): Promise<boolean> {
  if (!store) return false
  try {
    if (!(await store.has('transformers-cache'))) return false
    return (await (await store.open('transformers-cache')).match(MODEL_FILE_URL)) !== undefined
  } catch { return false }
}
```

- [ ] **Step 4: Implement the components**

`src/app/LoadRealButton.tsx`:

```tsx
import { useRealModel } from './realModel'

export function LoadRealButton({ variant = 'button' }: { variant?: 'button' | 'link' }) {
  const request = useRealModel((s) => s.request)
  const loading = useRealModel((s) => s.loading)
  return (
    <button type="button" data-testid="load-real" className={variant === 'link' ? 'link-button' : 'load-real'}
      disabled={loading} onClick={request}>Load the real model</button>
  )
}
```

`src/app/ModelSlot.tsx`:

```tsx
import type { ProgressInfo } from '../engine/types'
import type { Mode } from '../trace/types'

// The one place for the model's status: the banner while simulated, the
// download progress while loading, a load error with its fallback.
export function ModelSlot({ mode, loading, fromCache, progress, error, bannerClosed, onLoad, onClose, onFallback }: {
  mode: Mode; loading: boolean; fromCache: boolean; progress: ProgressInfo | null; error: string | null
  bannerClosed: boolean; onLoad(): void; onClose(): void; onFallback(): void
}) {
  if (error) return (
    <div data-testid="model-error" className="model-slot model-error">
      Real model unavailable: {error}
      <button data-testid="btn-fallback" onClick={onFallback}>Continue in Simulated mode</button>
    </div>
  )
  if (loading) {
    const pct = progress && progress.total > 0 ? Math.round((progress.loaded / progress.total) * 100) : null
    return (
      <div data-testid="model-progress" className="model-slot model-progress">
        {fromCache ? 'Loading the real model from the browser’s cache…'
          : progress ? `Loading the real model: ${progress.file} · ${pct ?? 0} %` : 'Loading the real model…'}
        {pct !== null && <progress data-testid="model-progress-bar" max={100} value={pct} />}
      </div>
    )
  }
  if (mode !== 'sim' || bannerClosed) return null
  return (
    <div data-testid="sim-banner" className="model-slot sim-banner">
      <p><strong>You are watching a simulation.</strong> Its numbers are made up to show the shape of the pipeline.
        Load the real model to see what SmolLM2 actually computes: real attention, the MLP inside each layer, and
        answers in Chat mode. 250 MB, downloaded once, runs entirely in your browser.</p>
      <button type="button" data-testid="banner-load" className="load-real" onClick={onLoad}>Load the real model</button>
      <button type="button" className="sim-banner-close" aria-label="Close" onClick={onClose}>×</button>
    </div>
  )
}
```

`src/app/ModeBadge.tsx`:

```tsx
import type { Mode } from '../trace/types'

export function ModeBadge({ mode, loading, device, attentions }: {
  mode: Mode; loading: boolean; device: 'webgpu' | 'wasm' | null; attentions: boolean
}) {
  const text = mode === 'sim' ? 'Simulated' : loading || !device ? 'Loading…' : `Real model · ${device}${attentions ? ' · attn' : ''}`
  return <span data-testid="mode-badge" className={`mode-badge mode-badge-${mode === 'sim' ? 'sim' : 'real'}`}>{text}</span>
}
```

`src/index.css` (replace `.device-chip` and `.model-status-slot`):

```css
.mode-badge { display: inline-block; font-size: .68rem; letter-spacing: .04em; text-transform: uppercase;
  border: 1px solid var(--line-strong); border-radius: 3px; padding: 0 .35rem; margin-left: .5rem; vertical-align: .2rem;
  color: var(--ink-soft); background: var(--card); }
.mode-badge-sim { border-color: var(--shu-soft); color: var(--shu); }
.model-slot { margin-top: .6rem; border-radius: 8px; padding: .55rem .8rem; font-size: .85rem; }
.sim-banner { display: flex; gap: .8rem; align-items: center; border: 1px solid var(--shu-soft); background: var(--shu-wash); }
.sim-banner p { flex: 1; margin: 0; line-height: 1.45; }
.sim-banner-close { border: 0; background: none; color: var(--ink-faint); font-size: 1.1rem; cursor: pointer; }
.model-progress { display: flex; gap: .6rem; align-items: center; border: 1px solid var(--line); background: var(--card); color: var(--ink-soft); }
.model-progress progress { flex: 1; max-width: 16rem; height: .4rem; accent-color: var(--ai); }
.load-real { border: 1px solid var(--ai); background: var(--ai); color: #fff; border-radius: 4px; padding: .3rem .7rem; white-space: nowrap; }
.load-real:disabled { opacity: .5; }
```

(Keep the existing `.model-error` rule and add `model-slot` spacing to it.)

- [ ] **Step 5: Wire up the app and the cards**

`src/App.tsx`:
- State: `const [bannerClosed, setBannerClosed] = useState(() => readFlag('tsumugi.simBannerClosed'))`, `const [fromCache, setFromCache] = useState(false)`.
- `const loading = mode === 'real' && !realReady && !modelError`.
- In `handleModeChange`, after `setRealReady(true)` in both success paths: `writeFlag('tsumugi.realModelUsed', true)`; in `finally`: `setFromCache(false)`.
- Start-up effect:

```ts
  useEffect(() => {
    if (!readFlag('tsumugi.realModelUsed')) return
    let live = true
    void realModelCached().then((ok) => {
      if (!live) return
      if (ok) { setFromCache(true); void handleModeChange('real') }
      else { writeFlag('tsumugi.simBannerClosed', false); setBannerClosed(false) }
    })
    return () => { live = false }
    // runs once on start; handleModeChange is stable enough for a one-shot call
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
```

- Store wiring:

```ts
  useEffect(() => {
    useRealModel.setState({ request: () => { void handleModeChange('real') }, loading })
  })
```

- Header: `<h1>… Tsumugi <ModeBadge mode={mode} loading={loading} device={device} attentions={attn} /> <span className="app-subtitle">…</span></h1>`.
- Under `<PromptBar …/>` (which loses its `status` prop):

```tsx
      <ModelSlot mode={mode} loading={loading} fromCache={fromCache} progress={progress} error={modelError}
        bannerClosed={bannerClosed} onLoad={() => { void handleModeChange('real') }}
        onClose={() => { writeFlag('tsumugi.simBannerClosed', true); setBannerClosed(true) }}
        onFallback={() => { setModelError(null); setMode('sim') }} />
```

- Remove the `ModelStatus` import; delete `src/app/ModelStatus.tsx` and its test.

`src/app/PromptBar.tsx`: remove the `status` prop and the `model-status-slot` span.

`src/viz/details/LayersDetail.tsx`: `const noteAction = mode === 'sim' ? <LoadRealButton /> : undefined`, passed as `noteAction={noteAction}` to `EditSizeCard`, `LayerVotesCard`, `NeuronCard`, `MlpStepsCard`. In each of those four cards add the optional prop `noteAction?: ReactNode` and render it inside the note paragraph after the text: `<p data-testid="…-note" className="mlp-note">{note} {noteAction}</p>`.

`src/viz/AttentionHeatmap.tsx`: the illustrative branch of the note becomes
`<>Illustrative pattern (simulated): the real model would show the weights it computed. <LoadRealButton variant="link" /></>`.

- [ ] **Step 6: Run the tests**

Run: `npx vitest run`
Expected: PASS (whole suite: the old `ModelStatus` cases now live in `ModelSlot`/`ModeBadge`).

- [ ] **Step 7: Commit**

```bash
git add -A src
git commit -m "feat(real-model): banner, title badge and status slot; load buttons where the simulation stops; remembered choices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Browser test, docs, and verification against the real model and the mockups

**Files:**
- Create: `e2e/chat.spec.ts`
- Modify: `e2e/*.spec.ts` only if the banner or the moved controls break them; `docs/BACKLOG.md`

**Interfaces:**
- Consumes: `buildFixtureTrace({ chat: true, mode: 'real', mlp: true })` (Task 4), the run-file import flow used by `e2e/steps.spec.ts`.

- [ ] **Step 1: Write the browser test**

`e2e/chat.spec.ts`:

```ts
import { expect, test } from '@playwright/test'
import { buildFixtureTrace } from '../src/test/fixtures'

test('an imported chat run: folded strip, end of turn, and an attention matrix that fits its card', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.goto('/')
  await expect(page.getByTestId('sim-banner')).toBeVisible()
  await expect(page.getByTestId('mode-badge')).toHaveText('Simulated')
  await page.getByTestId('prompt-input').fill('seed run')
  await page.getByTestId('btn-generate').click()
  await expect(page.getByTestId('run-chip')).toHaveCount(1, { timeout: 15000 })

  const events = buildFixtureTrace({ chat: true, mode: 'real', mlp: true })
  const file = JSON.stringify({ formatVersion: 1, events, meta: { prompt: 'What is the capital of France?',
    params: { temperature: 0, topK: 10, maxNewTokens: 8 }, mode: 'real', modelId: 'fixture', endedAt: Date.now(),
    reason: 'eos', format: { kind: 'chat', system: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face' } } })
  await page.getByTestId('import-input').setInputFiles({ name: 'run.json', mimeType: 'application/json', buffer: Buffer.from(file) })
  const chip = page.getByTestId('run-chip').filter({ hasText: 'Chat · What is the' })
  await chip.getByTestId('run-chip-main').click()

  await expect(page.getByTestId('system-fold')).toHaveText('system prompt · 20 tokens ▸')
  await expect(page.getByTestId('generated-token').last()).toHaveClass(/token-end/)
  await expect(page.getByTestId('generated-token').filter({ hasText: 'Paris' })).toHaveCount(1)

  await page.getByTestId('stage-card').nth(2).click()
  const heatmap = page.getByTestId('attention-heatmap')
  const card = (await heatmap.boundingBox())!
  const svg = (await heatmap.locator('svg').boundingBox())!
  expect(svg.width).toBeLessThanOrEqual(card.width)
  await expect(page.getByTestId('attn-show-system')).not.toBeChecked()
  await expect(page.getByTestId('edit-fold')).toHaveText('system prompt · 20 tokens ▸')
})
```

- [ ] **Step 2: Run the browser tests**

Run: `npx playwright test`
Expected: all specs PASS. If an existing spec fails because the banner shifts the layout or a control moved, fix the spec (not the app) and note it in the commit message.

- [ ] **Step 3: Update the backlog**

In `docs/BACKLOG.md`, change `- [ ] **#8 Chat templating**` to `- [x]` and replace its description with:

```markdown
- [x] **#8 Chat templating** — real runs can go through SmolLM2's own chat
  template ("Chat", the default for typed prompts) or as plain text
  ("Continue text"), with an editable system prompt (empty = no system
  turn). The template's tokens are shown dashed, the system turn folds in
  the strip, the attention matrix and Edit sizes, and the attention matrix's
  cells fit the card (20 → 8 px). A "Chat: a question" example; the
  real model got a Simulated | Real switch, a banner, a title badge and
  load buttons where the simulation stops. Spec
  `docs/superpowers/specs/2026-10-21-chat-templating-design.md`.
```

- [ ] **Step 4: Full checks**

Run: `npm test && npx tsc -b && npm run lint && npm run build && npx playwright test`
Expected: all pass; lint shows only the 2 known warnings.

- [ ] **Step 5: Commit**

```bash
git add e2e docs/BACKLOG.md
git commit -m "test(chat): imported chat run in the browser; backlog #8 done

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Verify against the real model (manual, local)**

Start `npm run dev`, open the app in a browser (Playwright MCP), click "Load the real model" and wait for the badge to read "Real model · …". Set T to 0, click "Chat: a question". Expected: the answer contains "Paris" and the last generated token is `<|im_end|>` (the browser's 8-bit model agrees with the fp32 findings). Then switch to Continue text, type "What is the capital of France?" with T = 0 and Generate. Expected: the first and only generated token is `<|im_end|>`. Record both outcomes in the final report; if either differs, report it with the generated tokens instead of changing the hint.

- [ ] **Step 7: Compare with the mockups (manual)**

Screenshot the prompt bar in Chat mode, the token strip of the chat run, the attention matrix with the fold, the banner in simulated mode, and the title badge. Open the mockups (`.superpowers/brainstorm/59075-1792614513/content/chat-mode.html` option B, `attention-matrix.html` option B, `real-mode-hint.html` option A) and compare layout, order of controls, chip text, dashed styles and the banner side by side. Fix deviations (each with a test where one applies) before reporting done.

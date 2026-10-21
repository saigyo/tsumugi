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
// "<|im_start|> ass istant \n". hasSystem says whether the prompt was built with a
// system turn, so a typed <|im_start|> cannot pass for one. Anything else gives undefined, and the views
// show every position unfolded.
export function chatSpans(ids: number[], imStart: number, imEnd: number, hasSystem: boolean): ChatSpans | undefined {
  const starts = ids.flatMap((id, i) => (id === imStart ? [i] : []))
  if (starts[0] !== 0 || starts.length !== (hasSystem ? 3 : 2)) return undefined
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

import type { PromptExample } from './PromptBar'

// The one example that runs in Chat mode; it needs the real model.
export const CHAT_EXAMPLE: PromptExample = {
  id: 'chat-question',
  label: 'Chat: a question',
  prompt: 'What is the capital of France?',
  format: 'chat',
  hint: 'As Continue text this question usually ends at once: the model\'s first token is <|im_end|> in about 9 of 10 runs. In Chat mode the template\'s tokens tell it a user asked and the assistant should reply. It answers, and often ends its turn itself with <|im_end|>; with T = 0 it always does.',
}

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

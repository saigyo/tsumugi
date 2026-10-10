import { visibleToken } from '../../viz/tokenText'
import type { GlossaryEntry } from '../types'

const SHOWN = 12

export const token: GlossaryEntry = {
  id: 'token',
  title: 'Token',
  summary: 'A piece of text, often a word or part of a word, that the model reads and writes as one unit.',
  stage: 'tokenize',
  related: ['embedding', 'sampling'],
  body: () => (
    <>
      <p>The model never sees letters. The tokenizer splits text into pieces from a fixed vocabulary of 49 152
        entries and replaces each piece with its id. Common words are one token; rare words are split into
        several.</p>
      <pre className="gl-formula">{`"The cat sat"  →  The | ␣cat | ␣sat  →  three ids`}</pre>
      <p>A leading space belongs to the token, shown as ␣: “␣cat” and “cat” are different vocabulary entries.
        Some ids are control markers rather than text, such as <code>{'<|im_end|>'}</code>, which ends a turn.</p>
      <p>Generation works on ids too: each cycle produces one new id, which the tokenizer turns back into
        text.</p>
    </>
  ),
  inRun: (run) => run.tokens.length > 0 ? (
    <p>Your prompt became {run.tokens.length} tokens:{' '}
      {run.tokens.slice(0, SHOWN).map((t) => visibleToken(t.text)).join(' | ')}
      {run.tokens.length > SHOWN ? ' | …' : ''}.</p>
  ) : null,
}

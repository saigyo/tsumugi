import { expect, test } from 'vitest'
import { useChatFolds } from './chatFolds'

test('the two folds open and close independently, for one run’s spans only', () => {
  const a = { user: [3, 4] as [number, number] }, b = { user: [3, 4] as [number, number] }
  const { set } = useChatFolds.getState()
  set(a, 'matrix', true)
  set(a, 'edits', true)
  set(a, 'matrix', false)
  expect(useChatFolds.getState().open).toEqual({ spans: a, folds: ['edits'] })
  set(b, 'matrix', true)
  expect(useChatFolds.getState().open).toEqual({ spans: b, folds: ['matrix'] })
})

import { expect, test } from 'vitest'
import { SPACE_MARKER, markLeadingSpace } from './spaceMarker'

test('a leading space becomes the visible marker', () => {
  expect(markLeadingSpace(' cat')).toBe(`${SPACE_MARKER}cat`)
})

test('text without a leading space is unchanged', () => {
  expect(markLeadingSpace('cat')).toBe('cat')
  expect(markLeadingSpace('')).toBe('')
})

test('only the first space is marked', () => {
  expect(markLeadingSpace('  cat')).toBe(`${SPACE_MARKER} cat`)
})

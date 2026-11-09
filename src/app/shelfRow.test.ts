import { expect, test } from 'vitest'
import { shelfRow } from './shelfRow'

const r = Array.from({ length: 8 }, (_, i) => ({ id: `r${i + 1}` }))
const id = (x: { id: string }) => x.id

test('the five newest, newest first', () => {
  expect(shelfRow(r, []).row.map(id)).toEqual(['r8', 'r7', 'r6', 'r5', 'r4'])
  expect(shelfRow(r, []).hidden).toBe(3)
})
test('older kept runs stay in their place', () => {
  expect(shelfRow(r, ['r2']).row.map(id)).toEqual(['r8', 'r7', 'r6', 'r5', 'r4', 'r2'])
  expect(shelfRow(r, ['r2']).hidden).toBe(2)
  expect(shelfRow(r, ['r2', 'r1', 'r7']).row.map(id)).toEqual(['r8', 'r7', 'r6', 'r5', 'r4', 'r2', 'r1'])
})
test('unknown and empty ids are ignored', () => {
  expect(shelfRow(r.slice(0, 3), ['gone', null]).row.map(id)).toEqual(['r3', 'r2', 'r1'])
  expect(shelfRow(r.slice(0, 3), []).hidden).toBe(0)
})

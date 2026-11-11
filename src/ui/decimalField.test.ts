import { describe, expect, it } from 'vitest'
import { decimalValue, normaliseDecimal, stepDecimal } from './decimalField'

describe('decimalField', () => {
  it('normaliseDecimal turns a comma into a point and rejects anything else', () => {
    expect(normaliseDecimal('0,85')).toBe('0.85'); expect(normaliseDecimal(',5')).toBe('.5'); expect(normaliseDecimal('0.')).toBe('0.')
    expect(normaliseDecimal('')).toBe(''); expect(normaliseDecimal('abc')).toBeNull(); expect(normaliseDecimal('1.2.3')).toBeNull()
  })
  it('decimalValue returns the number, brought within the limits', () => {
    expect(decimalValue('0.85', 0, 2)).toBe(0.85); expect(decimalValue('.5', 0, 2)).toBe(0.5)
    expect(decimalValue('0.', 0, 2)).toBe(0); expect(decimalValue('', 0, 2)).toBeNull(); expect(decimalValue('2.5', 0, 2)).toBe(2); expect(decimalValue('.', 0, 2)).toBeNull()
  })
  it('stepDecimal steps by a tenth within the limits', () => {
    expect(stepDecimal(0.8, 0.1, 0, 2)).toBe(0.9); expect(stepDecimal(0.3, -0.1, 0, 2)).toBe(0.2); expect(stepDecimal(0.84, 0.1, 0, 2)).toBe(0.9)
    expect(stepDecimal(2, 0.1, 0, 2)).toBe(2); expect(stepDecimal(0, -0.1, 0, 2)).toBe(0)
  })
})

import { describe, expect, it } from 'vitest'
import { decimalValue, normaliseDecimal, stepDecimal } from './decimalField'

describe('decimalField', () => {
  it('normaliseDecimal turns a comma into a point and rejects anything else', () => {
    expect(normaliseDecimal('0,85')).toBe('0.85'); expect(normaliseDecimal(',5')).toBe('.5'); expect(normaliseDecimal('0.')).toBe('0.')
    expect(normaliseDecimal('')).toBe(''); expect(normaliseDecimal('abc')).toBeNull(); expect(normaliseDecimal('1.2.3')).toBeNull()
  })
  it('normaliseDecimal trims and maps full-width characters', () => {
    expect(normaliseDecimal(' 0.8 ')).toBe('0.8'); expect(normaliseDecimal('０．８５')).toBe('0.85'); expect(normaliseDecimal('０，８')).toBe('0.8')
    expect(normaliseDecimal('  ')).toBe(''); expect(normaliseDecimal('0 8')).toBeNull()
  })
  it('decimalValue returns the number, brought within the limits', () => {
    expect(decimalValue('0.85', 0, 2)).toBe(0.85); expect(decimalValue('.5', 0, 2)).toBe(0.5)
    expect(decimalValue('0.', 0, 2)).toBe(0); expect(decimalValue('', 0, 2)).toBeNull(); expect(decimalValue('2.5', 0, 2)).toBe(2); expect(decimalValue('.', 0, 2)).toBeNull()
  })
  it('stepDecimal steps by a tenth within the limits', () => {
    expect(stepDecimal(0.8, 0.1, 0, 2)).toBe(0.9); expect(stepDecimal(0.3, -0.1, 0, 2)).toBe(0.2); expect(stepDecimal(0.84, 0.1, 0, 2)).toBe(0.9)
    expect(stepDecimal(2, 0.1, 0, 2)).toBe(2); expect(stepDecimal(0, -0.1, 0, 2)).toBe(0)
  })
  it('stepDecimal goes to the next multiple of 0.1 in the direction asked, without float noise', () => {
    expect(stepDecimal(1.25, 0.1, 0, 2)).toBe(1.3); expect(stepDecimal(1.25, -0.1, 0, 2)).toBe(1.2)
    expect(stepDecimal(0.8, 0.1, 0, 2)).toBe(0.9); expect(stepDecimal(0.8, -0.1, 0, 2)).toBe(0.7)
    expect(stepDecimal(0.2, 0.1, 0, 2)).toBe(0.3); expect(stepDecimal(0.3, -0.1, 0, 2)).toBe(0.2)
    expect(stepDecimal(0.7, 0.1, 0, 2)).toBe(0.8); expect(stepDecimal(1.9, 0.1, 0, 2)).toBe(2); expect(stepDecimal(0.04, -0.1, 0, 2)).toBe(0)
    for (let v = 0; v <= 2; v += 0.1) expect(String(stepDecimal(v, 0.1, 0, 2))).toMatch(/^\d(\.\d)?$/)
  })
})

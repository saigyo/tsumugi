import { describe, expect, it } from 'vitest'
import { count, fixed, percent, points, signed } from './format'

describe('format', () => {
  it('count groups with a thin space from 10 000', () => {
    expect(count(49152)).toBe('49 152'); expect(count(1536)).toBe('1536'); expect(count(9999)).toBe('9999')
    expect(count(10000)).toBe('10 000'); expect(count(1234567)).toBe('1 234 567')
  })
  it('fixed uses a real minus and no negative zero', () => {
    expect(fixed(-0.514, 2)).toBe('−0.51'); expect(fixed(-0.001, 2)).toBe('0.00'); expect(fixed(16.36, 2)).toBe('16.36')
  })
  it('signed always carries a sign except zero', () => {
    expect(signed(4.29)).toBe('+4.29'); expect(signed(-0.15)).toBe('−0.15'); expect(signed(0)).toBe('0.00'); expect(signed(-0.004)).toBe('0.00')
    expect(signed(34.94, 1)).toBe('+34.9')
  })
  it('percent has no space and avoids false 0 and 100', () => {
    expect(percent(0.94)).toBe('94%'); expect(percent(0)).toBe('0%'); expect(percent(0.003)).toBe('<1%')
    expect(percent(1)).toBe('100%'); expect(percent(0.996)).toBe('>99%'); expect(percent(null)).toBe('—')
  })
  it('points shows differences in percentage points', () => {
    expect(points(0.63)).toBe('+63 pts'); expect(points(-0.79, true)).toBe('≈ −79 pts'); expect(points(0)).toBe('0 pts')
  })
})

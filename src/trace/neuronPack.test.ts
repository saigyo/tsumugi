import { expect, test } from 'vitest'
import { packFloat32, packHalf, unpackFloat32, unpackHalf } from './neuronPack'

test('float16 round trip stays within half-precision rounding', () => {
  const values = [0, 1, -1, 0.1, -28.25, 24.36, 3.59e-3, 6.1e-5, 1e-7, 65504, -0.0004]
  const back = unpackHalf(packHalf(values))
  expect(back).toHaveLength(values.length)
  values.forEach((v, i) => {
    const tol = Math.abs(v) >= 6.1e-5 ? Math.abs(v) * 2 ** -11 : 2 ** -25
    expect(Math.abs(back[i] - v)).toBeLessThanOrEqual(tol)
  })
})

test('float16 saturates to infinity beyond its range and keeps the sign of zero', () => {
  const back = unpackHalf(packHalf([1e6, -1e6, -0]))
  expect(back[0]).toBe(Infinity)
  expect(back[1]).toBe(-Infinity)
  expect(Object.is(back[2], -0)).toBe(true)
})

test('a long array survives (chunked base64)', () => {
  const values = Array.from({ length: 46080 }, (_, i) => Math.sin(i) * 20)
  const back = unpackHalf(packHalf(values))
  expect(back).toHaveLength(46080)
  expect(Math.abs(back[12345] - values[12345])).toBeLessThan(0.02)
})

test('float32 round trip is exact', () => {
  const values = Float32Array.from([0.123456789, -3.5, 1e-30])
  expect([...unpackFloat32(packFloat32(values))]).toEqual([...values])
})

test('malformed data throws', () => {
  expect(() => unpackHalf('###')).toThrow()
  expect(() => unpackHalf(btoa('abc'))).toThrow(/odd/)
  expect(() => unpackFloat32(btoa('abcdef'))).toThrow(/multiple of 4/)
})

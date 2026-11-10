import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import { THEME_KEY, initTheme, nextMode, readMode, resolveTheme, useTheme } from './theme'

function mockMedia(initial: boolean) {
  let listener: ((e: { matches: boolean }) => void) | null = null
  const mql = {
    matches: initial,
    addEventListener: (_: string, l: (e: { matches: boolean }) => void) => { listener = l },
    removeEventListener: () => { listener = null },
  }
  vi.stubGlobal('matchMedia', () => mql)
  return { fire: (matches: boolean) => { mql.matches = matches; listener?.({ matches }) } }
}

beforeEach(() => { localStorage.clear(); delete document.documentElement.dataset.theme })
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

describe('readMode', () => {
  test('defaults to auto and accepts light and dark', () => {
    expect(readMode()).toBe('auto')
    localStorage.setItem(THEME_KEY, 'dark'); expect(readMode()).toBe('dark')
    localStorage.setItem(THEME_KEY, 'light'); expect(readMode()).toBe('light')
  })
  test('ignores unknown values and throwing storage', () => {
    localStorage.setItem(THEME_KEY, 'purple'); expect(readMode()).toBe('auto')
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked') })
    expect(readMode()).toBe('auto')
  })
})

test('resolveTheme', () => {
  expect(resolveTheme('auto', true)).toBe('dark')
  expect(resolveTheme('auto', false)).toBe('light')
  expect(resolveTheme('light', true)).toBe('light')
  expect(resolveTheme('dark', false)).toBe('dark')
})

test('nextMode cycles auto, light, dark', () => {
  expect(nextMode('auto')).toBe('light')
  expect(nextMode('light')).toBe('dark')
  expect(nextMode('dark')).toBe('auto')
})

describe('initTheme', () => {
  test('follows the system in auto, and stops following once a mode is chosen', () => {
    const media = mockMedia(true)
    const cleanup = initTheme()
    expect(document.documentElement.dataset.theme).toBe('dark')
    media.fire(false)
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(useTheme.getState().theme).toBe('light')
    useTheme.getState().cycle() // light
    useTheme.getState().cycle() // dark
    media.fire(false)
    expect(document.documentElement.dataset.theme).toBe('dark')
    cleanup()
  })

  test('cycle stores light and dark, and removes the key on returning to auto', () => {
    mockMedia(false)
    const cleanup = initTheme()
    useTheme.getState().cycle(); expect(localStorage.getItem(THEME_KEY)).toBe('light')
    useTheme.getState().cycle(); expect(localStorage.getItem(THEME_KEY)).toBe('dark')
    useTheme.getState().cycle(); expect(localStorage.getItem(THEME_KEY)).toBeNull()
    expect(useTheme.getState().mode).toBe('auto')
    cleanup()
  })

  test('cycle still works when storage throws', () => {
    mockMedia(false)
    const cleanup = initTheme()
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked') })
    useTheme.getState().cycle()
    expect(useTheme.getState().mode).toBe('light')
    useTheme.getState().cycle()
    expect(useTheme.getState().mode).toBe('dark')
    expect(document.documentElement.dataset.theme).toBe('dark')
    cleanup()
  })
})

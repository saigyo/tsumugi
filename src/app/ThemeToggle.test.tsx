import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, expect, test } from 'vitest'
import { ThemeToggle } from './ThemeToggle'
import { useTheme } from './theme'

beforeEach(() => { localStorage.clear(); useTheme.setState({ mode: 'auto', theme: 'light' }) })

test('the switch cycles auto, light, dark, auto with matching name, title and glyph', () => {
  render(<ThemeToggle />)
  const btn = screen.getByTestId('theme-toggle')
  const expectState = (label: string, glyph: string) => {
    expect(btn.getAttribute('aria-label')).toBe(label)
    expect(btn.getAttribute('title')).toBe(label)
    expect(btn.textContent).toBe(glyph)
  }
  expectState('Theme: auto (follows system)', '◐')
  fireEvent.click(btn); expectState('Theme: light', '☀')
  fireEvent.click(btn); expectState('Theme: dark', '☾')
  fireEvent.click(btn); expectState('Theme: auto (follows system)', '◐')
})

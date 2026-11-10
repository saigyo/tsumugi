import { useTheme, type ThemeMode } from './theme'

const LOOK: Record<ThemeMode, { glyph: string; label: string }> = {
  auto: { glyph: '◐', label: 'Theme: auto (follows system)' },
  light: { glyph: '☀', label: 'Theme: light' },
  dark: { glyph: '☾', label: 'Theme: dark' },
}

// Header switch: Auto → Light → Dark → Auto.
export function ThemeToggle() {
  const mode = useTheme((s) => s.mode)
  const cycle = useTheme((s) => s.cycle)
  const { glyph, label } = LOOK[mode]
  return (
    <button type="button" data-testid="theme-toggle" className="hdr-btn" aria-label={label} title={label}
      onClick={cycle}>{glyph}</button>
  )
}

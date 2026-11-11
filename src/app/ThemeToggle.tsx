import { Icon, type IconName } from '../ui/Icon'
import { useTheme, type ThemeMode } from './theme'

const LOOK: Record<ThemeMode, { icon: IconName; label: string }> = {
  auto: { icon: 'theme-auto', label: 'Theme: auto (follows system)' },
  light: { icon: 'theme-light', label: 'Theme: light' },
  dark: { icon: 'theme-dark', label: 'Theme: dark' },
}

// Header switch: Auto → Light → Dark → Auto.
export function ThemeToggle() {
  const mode = useTheme((s) => s.mode)
  const cycle = useTheme((s) => s.cycle)
  const { icon, label } = LOOK[mode]
  return (
    <button type="button" data-testid="theme-toggle" className="btn btn-icon" aria-label={label} title={label}
      onClick={cycle}><Icon name={icon} /></button>
  )
}

import { create } from 'zustand'

export type ThemeMode = 'auto' | 'light' | 'dark'
export type Theme = 'light' | 'dark'
export const THEME_KEY = 'tsumugi.theme'

// The same stored-choice / system-setting logic is repeated by necessity in the inline script in index.html,
// which must run before the app loads; keep the two in step.
const QUERY = '(prefers-color-scheme: dark)'

export function readMode(): ThemeMode {
  try {
    const v = localStorage.getItem(THEME_KEY)
    return v === 'light' || v === 'dark' ? v : 'auto'
  } catch { return 'auto' }
}
function writeMode(mode: ThemeMode): void {
  try {
    if (mode === 'auto') localStorage.removeItem(THEME_KEY)
    else localStorage.setItem(THEME_KEY, mode)
  } catch { /* storage unavailable: the choice lasts for this page only */ }
}

export function resolveTheme(mode: ThemeMode, systemDark: boolean): Theme {
  return mode === 'auto' ? (systemDark ? 'dark' : 'light') : mode
}
export function nextMode(mode: ThemeMode): ThemeMode {
  return mode === 'auto' ? 'light' : mode === 'light' ? 'dark' : 'auto'
}

function systemQuery(): MediaQueryList | null {
  try { return typeof window.matchMedia === 'function' ? window.matchMedia(QUERY) : null } catch { return null }
}
const systemDark = () => systemQuery()?.matches ?? false
const apply = (theme: Theme) => { document.documentElement.dataset.theme = theme }

interface ThemeState { mode: ThemeMode; theme: Theme; cycle(): void }

const initialMode = readMode()
export const useTheme = create<ThemeState>()((set, get) => ({
  mode: initialMode,
  theme: resolveTheme(initialMode, systemDark()),
  cycle: () => {
    const mode = nextMode(get().mode)
    const theme = resolveTheme(mode, systemDark())
    writeMode(mode)
    apply(theme)
    set({ mode, theme })
  },
}))

// Sets data-theme from the stored choice and the system setting, and follows
// live system changes while the mode is Auto. The inline script in index.html
// has normally set the attribute already; this keeps the store in step.
export function initTheme(): () => void {
  const query = systemQuery()
  const mode = readMode()
  const theme = resolveTheme(mode, query?.matches ?? false)
  apply(theme)
  useTheme.setState({ mode, theme })
  if (!query) return () => {}
  const onChange = (e: { matches: boolean }) => {
    if (useTheme.getState().mode !== 'auto') return
    const next = resolveTheme('auto', e.matches)
    apply(next)
    useTheme.setState({ theme: next })
  }
  query.addEventListener('change', onChange)
  return () => query.removeEventListener('change', onChange)
}

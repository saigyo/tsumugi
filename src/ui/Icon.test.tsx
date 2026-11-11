import { render } from '@testing-library/react'
import { expect, test } from 'vitest'
import { Icon, type IconName } from './Icon'

const NAMES: IconName[] = ['pin', 'pinned', 'link', 'download', 'close', 'import', 'compare', 'theme-auto', 'theme-light', 'theme-dark',
  'info', 'chevron', 'first', 'back', 'play', 'pause', 'forward', 'last', 'live', 'prev', 'next']

test.each(NAMES)('%s renders a hidden svg icon with a shape', (name) => {
  const { container } = render(<Icon name={name} />)
  const svg = container.querySelector('svg.icon')
  expect(svg).not.toBeNull()
  expect(svg?.getAttribute('aria-hidden')).toBe('true')
  expect(svg?.getAttribute('viewBox')).toBe('0 0 16 16')
  expect(svg?.querySelector('path, circle')).not.toBeNull()
})

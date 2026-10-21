import { render, screen, cleanup } from '@testing-library/react'
import { afterEach, expect, test } from 'vitest'
import { ModeBadge } from './ModeBadge'

afterEach(cleanup)

test('badge texts for simulated, loading and ready', () => {
  render(<ModeBadge mode="sim" loading={false} device={null} attentions={false} />)
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Simulated')
  cleanup()
  render(<ModeBadge mode="real" loading device={null} attentions={false} />)
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Loading…')
  cleanup()
  render(<ModeBadge mode="real" loading={false} device="webgpu" attentions />)
  expect(screen.getByTestId('mode-badge')).toHaveTextContent('Real model · webgpu · attn')
})

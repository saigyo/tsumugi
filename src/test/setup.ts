import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

// Unit tests never touch the network: any un-stubbed fetch rejects immediately.
// Tests that need fetch install their own stub with vi.stubGlobal('fetch', …).
vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in unit tests'))))

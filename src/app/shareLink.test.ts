import { describe, expect, it } from 'vitest'
import type { TraceEvent } from '../trace/types'
import { TEXT_FORMAT } from '../engine/chatTemplate'
import { buildShareLink, continueLimit, parseSeed, parseShareLink, shareableRun } from './shareLink'
import type { ShareSettings } from './shareLink'

const BASE = 'https://example.org/tsumugi/'
const params = { temperature: 0.8, topK: 10, maxNewTokens: 20 }
const hashOf = (url: string) => url.slice(url.indexOf('#'))

describe('share link', () => {
  it('round-trips text and chat settings', () => {
    const chat: ShareSettings = { prompt: 'hello', format: { kind: 'chat', system: 'Be brief.' }, params, seed: 123456789, rev: 'v7' }
    const link = buildShareLink(BASE, chat)
    expect(link.startsWith(BASE + '#run=1&')).toBe(true)
    expect(parseShareLink(hashOf(link))).toEqual(chat)
    const text: ShareSettings = { prompt: 'hello', format: TEXT_FORMAT, params, seed: 5, rev: 'v7' }
    const textLink = buildShareLink(BASE, text)
    expect(new URLSearchParams(hashOf(textLink).slice(1)).has('s')).toBe(false)
    expect(parseShareLink(hashOf(textLink))).toEqual(text)
  })

  it('round-trips a chat without a system turn', () => {
    const s: ShareSettings = { prompt: 'hi', format: { kind: 'chat', system: null }, params, seed: 1 }
    expect(parseShareLink(hashOf(buildShareLink(BASE, s)))).toEqual(s)
  })

  it('round-trips awkward prompts', () => {
    const s: ShareSettings = { prompt: '紬 & co #1\n= x', format: TEXT_FORMAT, params, seed: 0 }
    expect(parseShareLink(hashOf(buildShareLink(BASE, s)))?.prompt).toBe(s.prompt)
  })

  it('omits rev for the stock model', () => {
    const link = buildShareLink(BASE, { prompt: 'a', format: TEXT_FORMAT, params, seed: 1 })
    expect(link).not.toContain('rev=')
    expect(parseShareLink(hashOf(link))?.rev).toBeUndefined()
  })

  it('returns null without run=1', () => {
    expect(parseShareLink('')).toBeNull()
    expect(parseShareLink('#foo')).toBeNull()
  })

  it('bad values fall back', () => {
    expect(parseShareLink('#run=1&p=hi&t=9&k=0&n=abc&seed=-1')).toEqual({ prompt: 'hi' })
  })
})

describe('continueLimit', () => {
  it('doubles up to 100', () => {
    expect(continueLimit(20)).toBe(40)
    expect(continueLimit(60)).toBe(100)
    expect(continueLimit(100)).toBeNull()
  })
})

const start = (extra: object = {}): TraceEvent =>
  ({ type: 'run-start', prompt: 'a', mode: 'real', modelId: 'm', params, seed: 42, revision: 'v7', ...extra }) as TraceEvent
const end = (reason: 'eos' | 'max-tokens' | 'aborted' | 'error'): TraceEvent => ({ type: 'run-end', reason })

describe('shareableRun', () => {
  it('gives the settings of a seeded real run that finished', () => {
    const expected = { prompt: 'a', format: TEXT_FORMAT, params, seed: 42, rev: 'v7' }
    expect(shareableRun([start(), end('eos')])).toEqual(expected)
    expect(shareableRun([start(), end('max-tokens')])).toEqual(expected)
    expect(shareableRun([start({ format: { kind: 'chat', system: 'S' } }), end('eos')])?.format).toEqual({ kind: 'chat', system: 'S' })
  })

  it('gives null otherwise', () => {
    expect(shareableRun([start({ mode: 'sim' }), end('eos')])).toBeNull()
    expect(shareableRun([start({ seed: undefined }), end('eos')])).toBeNull()
    expect(shareableRun([start({ knockout: { layer: 1, cycle: 0, sourceId: 'x' } }), end('eos')])).toBeNull()
    expect(shareableRun([start(), end('aborted')])).toBeNull()
    expect(shareableRun([start(), end('error')])).toBeNull()
    expect(shareableRun([start()])).toBeNull()
  })

  it('omits rev when the run had none', () => {
    expect(shareableRun([start({ revision: undefined }), end('eos')])).not.toHaveProperty('rev')
  })
})

describe('parseSeed', () => {
  it('parses, treats blank as random and rejects the rest', () => {
    expect(parseSeed('')).toBeNull()
    expect(parseSeed('   ')).toBeNull()
    expect(parseSeed(' 42 ')).toBe(42)
    expect(parseSeed('4294967295')).toBe(4294967295)
    for (const bad of ['4294967296', '-1', '1.5', 'x']) expect(parseSeed(bad)).toBeUndefined()
  })
})

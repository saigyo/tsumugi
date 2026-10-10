import { afterEach, expect, test } from 'vitest'
import { resetGlossaryStore, useGlossaryStore } from './store'

afterEach(() => { resetGlossaryStore(); document.body.innerHTML = '' })
const s = () => useGlossaryStore.getState()

test('opening an entry from closed sets the view and remembers the opener', () => {
  const btn = document.createElement('button')
  s().openEntry('edit', undefined, btn)
  expect(s().view).toEqual({ kind: 'entry', id: 'edit' })
  expect(s().history).toEqual([])
  expect(s().opener).toBe(btn)
})

test('opening another entry pushes the previous view; back pops it', () => {
  s().openIndex()
  s().openEntry('softmax', 'temperature')
  expect(s().history).toEqual([{ kind: 'index' }])
  s().openEntry('logit')
  expect(s().history).toHaveLength(2)
  s().back()
  expect(s().view).toEqual({ kind: 'entry', id: 'softmax', section: 'temperature' })
  s().back()
  expect(s().view).toEqual({ kind: 'index' })
  s().back()                                   // empty history: stays put
  expect(s().view).toEqual({ kind: 'index' })
})

test('navigating inside the drawer keeps the original opener', () => {
  const first = document.createElement('button')
  const drawer = document.createElement('aside')
  drawer.className = 'gl-drawer'
  const inside = document.createElement('button')
  drawer.append(inside)
  document.body.append(first, drawer)
  s().openEntry('edit', undefined, first)
  s().openEntry('mlp', undefined, inside)
  expect(s().opener).toBe(first)
})

test('a term on the page clicked while the drawer is open becomes the new opener', () => {
  const first = document.createElement('button')
  const second = document.createElement('button')
  document.body.append(first, second)
  s().openEntry('embedding', 'learned', first)
  s().openEntry('softmax', 'temperature', second)
  expect(s().opener).toBe(second)
  s().close()
  expect(document.activeElement).toBe(second)
})

test('opening the open view again is a no-op', () => {
  s().openEntry('edit')
  s().openEntry('edit')
  expect(s().history).toEqual([])
  s().openIndex()
  s().openIndex()
  expect(s().history).toEqual([{ kind: 'entry', id: 'edit' }])
})

test('close clears view and history and returns focus to the opener', () => {
  const btn = document.createElement('button')
  document.body.append(btn)
  s().openEntry('edit', undefined, btn)
  s().openEntry('mlp')
  s().close()
  expect(s().view).toBeNull()
  expect(s().history).toEqual([])
  expect(s().opener).toBeNull()
  expect(document.activeElement).toBe(btn)
})

test('close with a detached opener focuses the help button', () => {
  const help = document.createElement('button')
  help.dataset.testid = 'btn-help'
  document.body.append(help)
  const gone = document.createElement('button')   // never attached: playback moved on
  s().openEntry('edit', undefined, gone)
  expect(() => s().close()).not.toThrow()
  expect(document.activeElement).toBe(help)
})

test('closing with the pointer returns focus quietly (no outline); with the keyboard it stays visible', () => {
  const btn = document.createElement('button')
  document.body.append(btn)
  s().openEntry('edit', undefined, btn)
  s().close('pointer')
  expect(document.activeElement).toBe(btn)
  expect(btn).toHaveAttribute('data-quiet-focus')
  btn.blur()
  expect(btn).not.toHaveAttribute('data-quiet-focus')
  s().openEntry('edit', undefined, btn)
  s().close('keyboard')
  expect(document.activeElement).toBe(btn)
  expect(btn).not.toHaveAttribute('data-quiet-focus')
})

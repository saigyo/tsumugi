import { expect, type Page } from '@playwright/test'
import { importChatRun, openFixtureRun } from '../fixtureRun'

export const THEMES = ['light', 'dark'] as const
export type Theme = (typeof THEMES)[number]

/** Emulates the system setting with no stored choice, then waits for colour transitions to finish. */
export async function setTheme(page: Page, theme: Theme): Promise<void> {
  await page.emulateMedia({ colorScheme: theme })
  await page.waitForTimeout(700)
}

async function settle(page: Page) {
  // the pause button exists only while playback runs
  const pause = page.getByTestId('btn-pause')
  if (await pause.count()) await pause.first().click()
  await page.waitForTimeout(450)
}

/** Empties the run archive and local storage, so every walk starts like a first visit. The deletion runs on a blank page of the same origin, where no connection of the app can block it. */
async function forgetEverything(page: Page) {
  await page.route('**/__blank', (route) => route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>blank</title>' }))
  await page.goto('/__blank')
  await page.evaluate(() => new Promise<void>((resolve, reject) => {
    localStorage.clear()
    sessionStorage.clear()
    const req = indexedDB.deleteDatabase('tsumugi')
    req.onsuccess = () => resolve()
    req.onerror = () => reject(req.error)
    req.onblocked = () => reject(new Error('the run archive is still open'))
  }))
  await page.unroute('**/__blank')
}

async function newestRun(page: Page): Promise<number> {
  const t = await page.getByTestId('run-chip').first().textContent({ timeout: 1000 }).catch(() => null)
  return t ? Number(t.match(/#(\d+)/)?.[1]) : 0
}

async function runOnce(page: Page, prompt: string) {
  const before = await newestRun(page)
  await page.getByTestId('prompt-input').fill(prompt)
  await page.getByTestId('btn-generate').click()
  await page.waitForFunction((n) => {
    const t = document.querySelector('[data-testid="run-chip"]')?.textContent
    return !!t && Number(t.match(/#(\d+)/)![1]) > n
  }, before, { timeout: 60_000 })
}

/** Drives the app and calls visit at each state, in a fixed order. Playback is paused before every visit. */
export async function walk(page: Page, visit: (state: string) => Promise<void>): Promise<void> {
  // `proof` asserts what makes the state that state, so a walk that lost its way fails here instead of visiting the wrong page
  const at = async (state: string, proof: () => Promise<void>) => {
    await settle(page)
    await proof()
    await visit(state)
  }
  const tipShowing = async () => { await expect(page.getByTestId('gl-tip').first()).toBeVisible() }
  const stageActive = (i: number) => async () => { await expect(stage(i)).toHaveAttribute('data-active', 'true') }
  const stage = (i: number) => page.getByTestId('stage-card').nth(i)

  await forgetEverything(page)
  await page.goto('/')
  await page.getByTestId('example-chip').first().waitFor()
  await page.waitForTimeout(500)
  await at('first-visit', async () => {
    await expect(page.getByTestId('example-chip').first()).toBeVisible()
    await expect(page.getByTestId('run-chip')).toHaveCount(0)
  })

  await page.getByTestId('example-chip').first().click()
  await page.getByTestId('run-chip').first().waitFor({ timeout: 60_000 })
  await page.getByTestId('btn-live').click()
  await at('sim-run-end', async () => { await expect(page.getByTestId('run-chip')).toHaveCount(1) })

  for (const [i, name] of [[1, 'embeddings'], [2, 'layers'], [3, 'logits'], [4, 'softmax'], [5, 'sampler']] as const) {
    await stage(i).click()
    await page.waitForTimeout(300)
    await at(`sim-${name}`, stageActive(i))
  }

  await page.evaluate(() => scrollTo(0, 0))
  await page.getByTestId('setting-info').first().hover()
  await page.waitForTimeout(400)
  await at('settings-tip', tipShowing)
  await page.mouse.move(2, 2)

  // the first stage whose detail card holds a glossary term
  for (const i of [3, 2, 1, 4, 0]) {
    await stage(i).click()
    await page.waitForTimeout(300)
    if (await page.locator('.detail .gl-term').count()) break
  }
  const term = page.locator('.detail .gl-term').first()
  await term.scrollIntoViewIfNeeded()
  await term.hover()
  await page.waitForTimeout(400)
  await at('term-tip', tipShowing)
  await page.mouse.move(2, 2)

  await page.evaluate(() => scrollTo(0, 0))
  await page.getByTestId('btn-help').first().click()
  await page.getByTestId('glossary-drawer').waitFor()
  await page.waitForTimeout(400)
  await at('glossary-index', async () => {
    await expect(page.getByTestId('glossary-drawer')).toBeVisible()
    await expect(page.getByTestId('gl-heading')).toHaveText('Glossary')
  })
  for (const entry of ['softmax', 'mlp']) {
    // a visit may have left the drawer closed or on an entry: get back to the index first
    if (!(await page.getByTestId('glossary-drawer').count())) await page.getByTestId('btn-help').first().click()
    const row = page.locator(`[data-testid="gl-index-row"][data-entry="${entry}"]`)
    if (!(await row.count())) await page.getByTestId('gl-all-entries').click()
    await row.click()
    await page.waitForTimeout(350)
    await at(`glossary-${entry}`, async () => {
      await expect(page.getByTestId('glossary-drawer')).toBeVisible()
      await expect(page.getByTestId('gl-heading')).toContainText(entry === 'mlp' ? /MLP/i : /softmax/i)
    })
  }
  if (await page.getByTestId('glossary-drawer').count()) await page.getByTestId('gl-close').click()
  await page.waitForTimeout(300)

  for (const p of ['The cat sat on the mat', 'Once upon a time', 'Paris is the capital of', 'Hello', 'The farmer bought a horse', '2 + 2 =', 'A very long prompt that keeps going to see how a run chip shortens its label']) {
    await runOnce(page, p)
  }
  await page.getByTestId('btn-live').click()
  const chip = page.getByTestId('run-chip').nth(2)
  await chip.hover() // the pin button shows on hover
  await chip.getByTestId('btn-chip-pin').click({ timeout: 5000 })
  await page.mouse.move(2, 2)
  await page.waitForTimeout(300)
  await at('runs-8', async () => {
    await expect(page.getByTestId('run-chip').first()).toContainText('#8')
    await expect(page.getByTestId('run-more')).toBeVisible()
  })

  await page.getByTestId('run-more').click()
  await page.getByTestId('run-list').waitFor()
  await page.waitForTimeout(300)
  await at('runs-list', async () => { await expect(page.getByTestId('run-list')).toBeVisible() })
  await page.keyboard.press('Escape')
  await page.waitForTimeout(200)

  await page.getByTestId('btn-compare-arm').click()
  await page.waitForTimeout(300)
  await at('compare-armed', async () => {
    await expect(page.getByTestId('btn-compare-exit')).toBeVisible()
    await expect(page.getByTestId('compare-view')).toHaveCount(0)
  })

  await page.getByTestId('run-chip-main').nth(1).click()
  await page.getByTestId('compare-view').waitFor({ timeout: 5000 })
  // the distribution switch appears only once a cycle is picked
  await page.getByTestId('cmp-tick').nth(2).click()
  await page.getByTestId('dist-mode-side').click()
  await page.waitForTimeout(400)
  await at('compare-side', async () => {
    await expect(page.getByTestId('compare-view')).toBeVisible()
    await expect(page.getByTestId('dist-mode-side')).toHaveAttribute('data-active', 'true')
  })
  await page.getByTestId('dist-mode-combined').click()
  await page.waitForTimeout(400)
  await at('compare-combined', async () => {
    await expect(page.getByTestId('dist-mode-combined')).toHaveAttribute('data-active', 'true')
  })
  await page.getByTestId('btn-compare-exit').click()

  await forgetEverything(page) // openFixtureRun expects an empty archive
  await openFixtureRun(page)
  await page.waitForTimeout(500)
  await at('real-layers', async () => {
    await stageActive(2)()
    await expect(page.getByTestId('run-chip').filter({ hasText: 'steps fixture' })).toHaveCount(1)
  })

  await importChatRun(page)
  await page.locator('[data-testid="stage-card"][data-stage="layers"]').click()
  await page.waitForTimeout(500)
  await at('chat-layers', async () => {
    await stageActive(2)()
    await expect(page.getByTestId('token-stream').getByTestId('system-fold')).toBeVisible()
    await expect(page.getByTestId('run-chip').filter({ hasText: 'What is the' })).toHaveCount(1)
  })
}

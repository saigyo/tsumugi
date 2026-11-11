import { readFileSync } from 'node:fs'
const d = JSON.parse(readFileSync(process.argv[2] + '/data.json', 'utf8'))
const what = process.argv[3]
if (what === 'hover') {
  const seen = new Set()
  for (const [label, rows] of Object.entries(d.hover)) for (const r of rows) {
    if (r.err) { console.log(`ERR  ${label} ${r.key} ${r.err}`); continue }
    if (!r.hov || !r.rest) continue
    const id = r.key
    if (seen.has(id)) continue; seen.add(id)
    const ch = []
    if (r.rest.bg !== r.hov.bg) ch.push(`bg ${r.rest.bg}→${r.hov.bg}`)
    if (r.rest.color !== r.hov.color) ch.push(`fg ${r.rest.color}→${r.hov.color}`)
    if (r.rest.bc !== r.hov.bc) ch.push(`bd ${r.rest.bc}→${r.hov.bc}`)
    if (r.rest.shadow !== r.hov.shadow) ch.push('shadow')
    if (r.rest.filter !== r.hov.filter) ch.push('filter ' + r.hov.filter)
    if (r.rest.opacity !== r.hov.opacity) ch.push('opacity ' + r.hov.opacity)
    if (r.rest.textDec !== r.hov.textDec) ch.push('textdec')
    const flag = (r.hov.cr !== null && r.hov.cr < 4.5 ? ` !!CR ${r.hov.cr}` : '') + (r.hov.shift > 0.5 ? ` !!SHIFT ${r.hov.shift} (${r.hov.moved})` : '')
    console.log(`${r.key.padEnd(70)} "${r.text}" | ${ch.join(', ') || 'NO CHANGE'}${flag} | rest cr ${r.rest.cr}`)
  }
}
if (what === 'inter' || what === 'boxes') {
  const m = new Map()
  for (const [label, rows] of Object.entries(d.inventory)) for (const r of rows) {
    if ((what === 'inter') !== r.inter) continue
    const id = `${r.key} h${r.h} fs${r.fs} fw${r.fw} r${r.rad} p[${r.pad}] bg${r.bg} fg${r.color} b${r.bw}${r.bs !== '-' ? ' ' + r.bs + ' ' + r.bc : ''} sh${r.shadow} tr[${r.trans}] ${r.cursor}`
    if (!m.has(id)) m.set(id, { r, labels: new Set() })
    m.get(id).labels.add(label)
  }
  const rows = [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  for (const [id, v] of rows) console.log(id + '  «' + v.r.text + '»  @' + [...v.labels].slice(0, 2).join(','))
}
if (what === 'text') {
  const m = new Map()
  for (const [label, rows] of Object.entries(d.text)) for (const r of rows) {
    const cur = m.get(r.id)
    if (!cur) m.set(r.id, { ...r, labels: new Set([label]) })
    else { cur.n += r.n; cur.labels.add(label); if (r.minCr < cur.minCr) { cur.minCr = r.minCr; cur.minEl = r.minEl; cur.minText = r.minText } for (const k of r.keys) if (!cur.keys.includes(k) && cur.keys.length < 5) cur.keys.push(k) }
  }
  for (const r of [...m.values()].sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))) console.log(`${r.id}  n=${r.n} minCR=${r.minCr}${r.minCr < 4.5 ? ' !! ' + r.minEl + ' "' + r.minText + '"' : ''}  [${r.keys.slice(0, 3).join(', ')}]`)
}
if (what === 'overflow') {
  for (const [label, o] of Object.entries(d.overflow)) {
    console.log(`== ${label} vw=${o.viewport} docScroll=${o.docScroll}`)
    if (o.past.length) console.log('  PAST: ' + o.past.join('\n        '))
    if (o.scrollers.length) console.log('  SCROLL: ' + o.scrollers.join(' ; '))
    if (o.truncated.length) console.log('  TRUNC: ' + o.truncated.slice(0, 6).join(' ; '))
  }
}
if (what === 'blocks') {
  for (const [label, b] of Object.entries(d.blocks)) {
    if (process.argv[4] && !label.includes(process.argv[4])) continue
    console.log(`== ${label}`)
    let prevBottom = null
    for (const r of b) { console.log(`  ${r.sel.padEnd(16)} x=${r.x} w=${r.w} y=${r.y} h=${r.h} gap=${prevBottom === null ? '-' : r.y - prevBottom} pad[${r.pad}] mt=${r.mt} mb=${r.mb}`); if (!['.mlp-card', '.gl-section', '.layers-nav'].includes(r.sel)) prevBottom = r.y + r.h }
  }
}
if (what === 'focus') for (const r of d.focus.top ?? []) console.log(`${r.el.padEnd(60)} outline=${r.outline} shadow=${r.shadow} parent=${r.parentShadow}`)

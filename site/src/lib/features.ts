// Mirror of src/r7b/features.py `handcrafted`. Both are checked against
// tests/fixtures/feature_examples.json, so keep them in lockstep.
import type { BrowserRouter } from './types'

const REASONING = new Set(['why', 'explain', 'prove', 'derive', 'compare', 'analyze', 'analyse',
  'calculate', 'estimate', 'design', 'implement', 'critique', 'justify'])
const CONSTRAINT = new Set(['exactly', 'only', 'must', 'without', 'under', 'least', 'most', 'words',
  'sentences', 'lines', 'rhyme', 'syllable', 'letter', 'format', 'formal'])
const SIMPLE = new Set(['what', 'who', 'when', 'where', 'name', 'list', 'which'])
const WH = new Set(['who', 'what', 'when', 'where', 'which', 'how', 'why'])
const CODE_CHARS = new Set('{}[]();=<>_')

const count = (re: RegExp, s: string) => (s.match(re) ?? []).length

export function handcrafted(text: string): number[] {
  const words = text.match(/[A-Za-z']+/g) ?? []
  const lower = words.map((w) => w.toLowerCase())
  const nChars = [...text].length
  const nWords = words.length
  const first = lower[0] ?? ''
  const chars = [...text]
  return [
    Math.log1p(nChars),
    Math.log1p(nWords),
    Math.log1p(count(/\n/g, text) + 1),
    Math.log1p(count(/\d+(?:[.,]\d+)?/g, text)),
    count(/^[A-D]\. /gm, text),
    count(/\?/g, text),
    nWords ? words.reduce((a, w) => a + w.length, 0) / nWords : 0,
    nWords ? words.filter((w) => w[0] === w[0].toUpperCase() && w[0] !== w[0].toLowerCase()).length / nWords : 0,
    Math.log1p(count(/[+\-*/=%^$×÷]/g, text)),
    nChars ? chars.filter((c) => CODE_CHARS.has(c)).length / nChars : 0,
    lower.filter((w) => REASONING.has(w)).length,
    lower.filter((w) => CONSTRAINT.has(w)).length,
    lower.filter((w) => SIMPLE.has(w)).length,
    WH.has(first) ? 1 : 0,
    Math.log1p(count(/[.!?](?:\s|$)/g, text)),
    count(/["\u201c\u201d]/g, text),
    text.includes('$') ? 1 : 0,
  ]
}

export interface Contribution {
  name: string
  value: number
  push: number
}

export function predict(router: BrowserRouter, text: string): { p: number; contributions: Contribution[] } {
  const x = handcrafted(text)
  let z = router.intercept
  const contributions = x.map((v, i) => {
    const push = router.coef[i] * ((v - router.mean[i]) / router.scale[i])
    z += push
    return { name: router.feature_names[i], value: v, push }
  })
  return { p: 1 / (1 + Math.exp(-z)), contributions }
}

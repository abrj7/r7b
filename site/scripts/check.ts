// Parity checks run with `npm test`:
// 1. the TypeScript feature port matches the Python fixture exactly;
// 2. recomputing the headline operating point in the browser code matches
//    what the Python pipeline reported for the test split.
import { readFileSync } from 'node:fs'
import { handcrafted } from '../src/lib/features.ts'
import { evaluateAt } from '../src/lib/routing.ts'
import type { QueryRow, Summary } from '../src/lib/types.ts'

let failures = 0
const close = (a: number, b: number, tol: number) => Math.abs(a - b) <= tol

const cases = JSON.parse(readFileSync(new URL('../../tests/fixtures/feature_examples.json', import.meta.url), 'utf8')) as {
  text: string
  features: number[]
}[]
for (const c of cases) {
  const got = handcrafted(c.text)
  got.forEach((v, i) => {
    if (!close(v, c.features[i], 1e-9)) {
      failures += 1
      console.error(`feature ${i} differs for ${JSON.stringify(c.text.slice(0, 40))}: ts=${v} py=${c.features[i]}`)
    }
  })
}
console.log(`features: ${cases.length} fixtures checked`)

const data = new URL('../public/data/', import.meta.url)
const summary = JSON.parse(readFileSync(new URL('summary.json', data), 'utf8')) as Summary
const queries = JSON.parse(readFileSync(new URL('queries.json', data), 'utf8')) as QueryRow[]
const test = queries.filter((q) => q.split === 'test')
const live = evaluateAt(test, summary.headline.threshold)
const want = summary.headline.test
for (const [k, got, exp] of [
  ['cost_saved', live.costSaved, want.cost_saved],
  ['quality_retained', live.qualityRetained, want.quality_retained],
  ['frac_strong', live.fracStrong, want.frac_strong],
] as const) {
  if (!close(got, exp, 1e-9)) {
    failures += 1
    console.error(`headline ${k}: browser=${got} python=${exp}`)
  }
}
console.log(`headline: recomputed on ${test.length} test queries`)

if (failures) {
  console.error(`${failures} mismatches`)
  process.exit(1)
}
console.log('ok')

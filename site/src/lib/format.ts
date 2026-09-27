import type { Curves, Run, Source } from './types'

export const pct = (x: number | null | undefined, digits = 1) =>
  x == null || Number.isNaN(x) ? '–' : `${(x * 100).toFixed(digits)}%`

export const num = (x: number | null | undefined, digits = 3) =>
  x == null || Number.isNaN(x) ? '–' : x.toFixed(digits)

export function usd(x: number): string {
  if (x === 0) return '$0'
  if (x < 0.001) return `$${(x * 1000).toFixed(3)}m`
  if (x < 1) return `$${x.toFixed(4)}`
  return `$${x.toFixed(2)}`
}

/** Dollars per 1,000 queries, the unit that makes per-query costs readable. */
export const perK = (x: number) => `$${(x * 1000).toFixed(x * 1000 < 1 ? 3 : 2)}`

export const SOURCE_LABEL: Record<Source, string> = {
  gsm8k: 'gsm8k',
  mmlu: 'mmlu',
  nq: 'nq-open',
  open: 'open-ended',
}

export const SOURCE_BLURB: Record<Source, string> = {
  gsm8k: 'grade-school math, exact numeric answer',
  mmlu: 'multiple choice across 57 subjects',
  nq: 'short factual questions, alias match',
  open: 'free-form, graded by a judge',
}

export const MODEL_LABEL: Record<Run['model'], string> = {
  logreg: 'logistic regression',
  gbdt: 'gradient boosting',
  mlp: 'small mlp',
}

export const FEATURE_LABEL: Record<Run['features'], string> = {
  handcrafted: 'hand-crafted',
  embed: 'embed v4',
  'embed+handcrafted': 'embed v4 + hand-crafted',
}

export const MODEL_COLOR: Record<Run['model'], string> = {
  logreg: 'var(--accent)',
  gbdt: 'var(--rust)',
  mlp: 'var(--purple)',
}

export function curveLabel(key: string, curves: Curves) {
  const c = curves.curves[key]
  const sameFeatures = c.features === curves.curves[curves.chosen].features
  return sameFeatures ? MODEL_LABEL[c.model] : `${MODEL_LABEL[c.model]}, ${FEATURE_LABEL[c.features]}`
}

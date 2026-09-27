// Browser twin of r7b.routing.evaluate_routing, run over precomputed per-query results.
import type { QueryRow, Source } from './types'

export interface Live {
  threshold: number
  n: number
  toR7b: number
  fracStrong: number
  cost: number
  quality: number
  costSaved: number
  qualityRetained: number
  strongCost: number
  strongQuality: number
  cheapQuality: number
  wrongEscalations: number
  missedEscalations: number
  bySource: Record<Source, { n: number; escalated: number }>
}

export function evaluateAt(rows: QueryRow[], threshold: number): Live {
  let cost = 0
  let quality = 0
  let strongCost = 0
  let strongQuality = 0
  let cheapQuality = 0
  let toR7b = 0
  let wrongEscalations = 0
  let missedEscalations = 0
  const bySource = {
    gsm8k: { n: 0, escalated: 0 },
    mmlu: { n: 0, escalated: 0 },
    nq: { n: 0, escalated: 0 },
    open: { n: 0, escalated: 0 },
  }
  for (const r of rows) {
    const cheap = r.p >= threshold
    cost += (cheap ? r.r7b.cost : r.command_a.cost) + r.router_cost
    quality += cheap ? r.r7b.quality : r.command_a.quality
    strongCost += r.command_a.cost
    strongQuality += r.command_a.quality
    cheapQuality += r.r7b.quality
    bySource[r.source].n += 1
    if (cheap) {
      toR7b += 1
      if (r.label === 0) missedEscalations += 1
    } else {
      bySource[r.source].escalated += 1
      if (r.label === 1) wrongEscalations += 1
    }
  }
  const n = rows.length || 1
  return {
    threshold,
    n: rows.length,
    toR7b,
    fracStrong: 1 - toR7b / n,
    cost,
    quality: quality / n,
    costSaved: strongCost ? 1 - cost / strongCost : 0,
    qualityRetained: strongQuality ? quality / strongQuality : 0,
    strongCost,
    strongQuality: strongQuality / n,
    cheapQuality: cheapQuality / n,
    wrongEscalations,
    missedEscalations,
    bySource,
  }
}

export function verdict(q: QueryRow, threshold: number) {
  const toR7b = q.p >= threshold
  if (toR7b && q.label === 0) return { toR7b, tag: 'missed', text: 'needed command a' } as const
  if (!toR7b && q.label === 1) return { toR7b, tag: 'overpaid', text: 'r7b would have done' } as const
  return { toR7b, tag: 'ok', text: 'right call' } as const
}

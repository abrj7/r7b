export type Source = 'gsm8k' | 'mmlu' | 'nq' | 'open'
export type Split = 'train' | 'val' | 'test'

export interface Point {
  threshold: number | null
  frac_strong: number
  cost: number
  quality: number
  cost_saved: number
  quality_retained: number
  pgr: number | null
}

export interface Run {
  model: 'logreg' | 'gbdt' | 'mlp'
  features: 'handcrafted' | 'embed' | 'embed+handcrafted'
  params: Record<string, unknown>
  cv_auc: number
  cv_auc_std: number
  val_auc: number
  val_apgr: number
  test_auc: number
  test_apgr: number
  test_brier: number
}

export interface OperatingPoint {
  target_retained: number
  threshold: number
  dev_oof: Point
  test: Point
  random_at_same_cost: Point
  random_quality_ci: [number, number]
  test_ci: { cost_saved: [number, number]; quality_retained: [number, number] }
}

interface ModelStats {
  quality: number
  input_tokens: number
  output_tokens: number
  cost_per_1k_queries: number
}

export interface SourceStats {
  n: number
  r7b_good_enough: number
  cheap: ModelStats
  strong: ModelStats
}

export interface JudgeValidation {
  n_pairs: number
  n_scores: number
  mae: number
  within_1: number
  spearman: number
  preference_agreement: number
  preference_kappa: number
  routing_label_agreement: number
  items: {
    id: string
    prompt_id: string
    human: [number, number]
    judge: [number, number]
    shown_swapped: boolean
    note: string
  }[]
}

export interface Summary {
  mode: 'mock' | 'live'
  generated_at: string
  models: { cheap: string; strong: string; judge: string; embed: string; embed_dim: number }
  pricing: {
    checked: string
    chat: Record<string, { input_per_m: number; output_per_m: number }>
    embed: Record<string, { input_per_m: number }>
  }
  budget: {
    queries: number
    open_queries: number
    judge_validation_items: number
    chat_calls: number
    judge_calls: number
    embed_calls: number
    total_calls: number
    monthly_cap: number
    reserve: number
    headroom: number
    calls_used_this_month: number | null
  }
  experiment_cost: {
    answers_usd: number
    judge_usd: number
    embed_usd: number
    total_usd: number
  }
  per_source: Record<Source | 'all', SourceStats>
  split_sizes: Record<Split, number>
  label_rate: Record<Split, number>
  runs: Run[]
  chosen: { model: Run['model']; features: Run['features']; params: Record<string, unknown> }
  headline: OperatingPoint
  operating_points: OperatingPoint[]
  baselines: { always_r7b: Point; always_command_a: Point; oracle: Point }
  judge_validation: JudgeValidation
}

export interface Curve {
  model: Run['model']
  features: Run['features']
  points: Point[]
  frontier: Point[]
}

export interface Curves {
  curves: Record<string, Curve>
  random_line: Point[]
  baselines: Summary['baselines']
  chosen: string
}

export interface Answer {
  text: string
  input_tokens: number
  output_tokens: number
  cost: number
  quality: number
  judge_score: number | null
}

export interface QueryRow {
  id: string
  source: Source
  split: Split
  text: string
  gold: string | string[] | null
  meta: { subject?: string; category?: string; difficulty?: number }
  label: 0 | 1
  p: number
  route: 'r7b' | 'command-a'
  router_cost: number
  r7b: Answer
  command_a: Answer
}

export interface BrowserRouter {
  feature_names: string[]
  mean: number[]
  scale: number[]
  coef: number[]
  intercept: number
  threshold: number
  cv_auc: number
  test_auc: number
  C: number
}

export interface SiteData {
  summary: Summary
  curves: Curves
  queries: QueryRow[]
  router: BrowserRouter
}

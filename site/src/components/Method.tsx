import type { ReactNode } from 'react'
import { num, pct, usd } from '@/lib/format'
import type { Summary } from '@/lib/types'
import { Section } from './Section'

function Part({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="block" style={{ marginTop: 'var(--s-7)' }}>
      <h3 className="mono" style={{ fontSize: 'var(--text-base)', fontWeight: 600, letterSpacing: '0.02em', marginBottom: 'var(--s-3)' }}>
        {title}
      </h3>
      <div className="method-body" style={{ display: 'grid', gap: 'var(--s-4)', color: 'var(--ink-soft)', maxWidth: '66ch' }}>
        {children}
      </div>
    </div>
  )
}

function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <div className="index" style={{ maxWidth: '34rem' }}>
      {rows.map(([k, v]) => (
        <div key={k} className="index__row" style={{ gridTemplateColumns: 'minmax(0,1fr) auto', padding: 'var(--s-2) 0' }}>
          <span style={{ fontSize: 'var(--text-sm)' }}>{k}</span>
          <span className="index__cell--num">{v}</span>
        </div>
      ))}
    </div>
  )
}

export function Method({ summary }: { summary: Summary }) {
  const s = summary
  const jv = s.judge_validation
  const b = s.budget
  const cheap = s.pricing.chat[s.models.cheap]
  const strong = s.pricing.chat[s.models.strong]
  const embed = s.pricing.embed[s.models.embed]
  const src = s.per_source

  return (
    <Section
      id="method"
      label="method"
      note="everything here is in the repo and reruns end to end"
      title="how it works"
      lede="the question is narrow on purpose: for one query, is the cheap model's answer as good as the expensive one's? everything else (the labels, the classifier, the curve) follows from answering that honestly on data where quality can be checked."
    >
      <Part title="queries">
        <p>
          {s.budget.queries} queries from four sources, sampled once with a fixed seed and committed:{' '}
          {src.gsm8k.n} from gsm8k (numeric answers), {src.mmlu.n} from mmlu (letter answers, all 57 subjects),{' '}
          {src.nq.n} from nq-open (short answers with aliases), and {src.open.n} open-ended prompts written for this
          project across writing, explanation, code, reasoning, planning, analysis, and knowledge. the first three have
          ground truth; the open-ended ones need a judge.
        </p>
      </Part>

      <Part title="answers and grading">
        <p>
          both {s.models.cheap} and {s.models.strong} get the identical message for every query, at temperature 0 with a
          fixed seed. ground-truth tasks ask for a final <code className="mono">Answer:</code> line: gsm8k is graded by
          numeric match, mmlu by the extracted letter, and nq-open by whether any gold alias appears as a whole-token span
          of the normalized answer.
        </p>
        <p>
          a query is labeled <strong>r7b is good enough</strong> when r7b's score is at least command a's. when both are
          wrong that still counts, since escalating would not have helped. for open-ended prompts, r7b must land within
          one rubric point of command a.
        </p>
      </Part>

      <Part title="the judge">
        <p>
          open-ended answers are scored 1–10 by {s.models.judge} against a written rubric (correctness, constraints,
          completeness, clarity, with hard caps for material errors and missed constraints). both answers go in one call,
          anonymized, with their order set by a hash of the query id, so neither model sits in the favored position
          systematically.
        </p>
        <p>
          before trusting it, the judge scored {jv.n_pairs} hand-labeled pairs ({jv.n_scores} answers) written with known
          flaws: a missed word limit, base-rate neglect, an informal register when formal was asked for, and so on.
        </p>
        <KV
          rows={[
            ['mean absolute error vs. human score', num(jv.mae, 2)],
            ['scores within one point', pct(jv.within_1, 0)],
            ['spearman correlation', num(jv.spearman, 2)],
            ['which-answer-is-better agreement', `${pct(jv.preference_agreement, 0)} (κ ${num(jv.preference_kappa, 2)})`],
            ['routing-label agreement', pct(jv.routing_label_agreement, 0)],
          ]}
        />
        {s.mode === 'mock' && (
          <p className="meta" style={{ lineHeight: 1.7 }}>
            in mock mode the judge is simulated as the human score plus noise, so these numbers only show the check runs;
            the live run replaces them.
          </p>
        )}
      </Part>

      <Part title="features and classifiers">
        <p>
          each query becomes a {s.models.embed_dim}-dimensional {s.models.embed} embedding (input type{' '}
          <code className="mono">classification</code>, batched 96 per request) plus 17 hand-crafted features: length,
          counts of numbers and math symbols, multiple-choice lines, code punctuation, reasoning verbs, constraint words,
          and so on. the router never sees which dataset a query came from.
        </p>
        <p>
          logistic regression, histogram gradient boosting, and a one-hidden-layer mlp are each trained on hand-crafted
          features, embeddings, and both. the split is {s.split_sizes.train} / {s.split_sizes.val} /{' '}
          {s.split_sizes.test}, stratified on source × label. hyperparameters come from 5-fold grid search on train, and
          the model is picked by cv auc. the threshold is the cheapest one that keeps the target share of command a
          quality on out-of-fold predictions over train + val. the chosen config is then refit on train + val and scored
          once on test.
        </p>
      </Part>

      <Part title="cost">
        <p>
          cost is billed tokens × list price, per query, using the token counts the api returns. the router's own
          overhead, one embed call per query, is included in every routed cost. judge calls are evaluation-only and not
          part of routing cost. prices were checked on {s.pricing.checked}:
        </p>
        <KV
          rows={[
            [`${s.models.cheap}, per 1m in / out`, `$${cheap.input_per_m} / $${cheap.output_per_m}`],
            [`${s.models.strong}, per 1m in / out`, `$${strong.input_per_m.toFixed(2)} / $${strong.output_per_m.toFixed(2)}`],
            [`${s.models.embed}, per 1m in`, `$${embed.input_per_m}`],
            ['whole experiment on a production key', usd(s.experiment_cost.total_usd)],
          ]}
        />
        <p className="meta" style={{ lineHeight: 1.7 }}>
          r7b and embed 4 are on cohere.com/pricing. command a no longer is (the page now leads with command a+);
          $2.50 / $10 is its list price and what cohere charges as a provider on openrouter.
        </p>
      </Part>

      <Part title="fitting in a free trial key">
        <p>
          trial keys allow 1,000 api calls a month, 20 chat requests a minute per model, and 2,000 embed inputs a minute.
          the dataset is sized to fit: every call is planned, written to a ledger before it is sent (so a crash still
          counts it), rate-limited per model, and cached to disk by request hash, so no request is ever paid for twice.
        </p>
        <KV
          rows={[
            [`answers (${b.queries} queries × 2 models)`, b.chat_calls],
            [`judge (${b.open_queries} open-ended + ${b.judge_validation_items} validation pairs)`, b.judge_calls],
            [`embeddings (${b.queries} texts, 96 per call)`, b.embed_calls],
            ['total planned', b.total_calls],
            [`monthly cap / reserve / headroom`, `${b.monthly_cap} / ${b.reserve} / ${b.headroom}`],
          ]}
        />
      </Part>

      <Part title="limitations">
        <p>
          <strong>small n.</strong> {s.split_sizes.test} test queries give wide intervals. the operating-point table shows
          them, and differences between routers inside a few auc points are not meaningful at this size. a production
          key, or a second month of trial calls, would let the same pipeline run on thousands.
        </p>
        <p>
          <strong>self-judging.</strong> command a grades its own answers on open-ended prompts. blinding, order
          randomization, and the validation slice limit the damage but can't rule out self-preference. a third-party
          judge would be the next step.
        </p>
        <p>
          <strong>the label is a choice.</strong> "good enough" means no worse than command a (one rubric point for
          open-ended). a looser tolerance saves more and a stricter one saves less; the curve covers every threshold but
          not every tolerance.
        </p>
        <p>
          <strong>one pair of models, one snapshot.</strong> prices and models move. command a+ now exists; the
          pipeline is model-agnostic, but these results are about r7b and command a.
        </p>
      </Part>
    </Section>
  )
}

import { FEATURE_LABEL, MODEL_LABEL, num, perK, pct, SOURCE_BLURB, SOURCE_LABEL } from '@/lib/format'
import type { Source, Summary } from '@/lib/types'
import { Section } from './Section'

export function Models({ summary }: { summary: Summary }) {
  const runs = [...summary.runs].sort((a, b) => b.cv_auc - a.cv_auc)
  const isChosen = (m: string, f: string) => summary.chosen.model === m && summary.chosen.features === f
  const cols = 'minmax(0,1.4fr) minmax(0,1.2fr) 6.5rem 4.2rem 4.2rem 4.2rem'
  const sources: Source[] = ['gsm8k', 'mmlu', 'nq', 'open']
  const srcCols = 'minmax(0,1.3fr) 3rem 4.6rem 4.6rem 4.6rem 5.5rem 5.5rem'

  return (
    <Section
      id="models"
      label="routers"
      note="three classifiers × three feature sets, 5-fold cv on train"
      title="which router, which features"
      lede="each classifier predicts one thing: will r7b's answer be as good as command a's? hyperparameters come from a 5-fold grid search on the 228 training queries; the winner is picked by mean cv auc, not by test. apgr is the average share of the r7b-to-command-a quality gap recovered across every escalation rate (random routing scores 0.5)."
    >
      <div style={{ overflowX: 'auto' }}>
        <div className="index" role="table" style={{ minWidth: 560 }}>
          <div className="index__head" role="row" style={{ gridTemplateColumns: cols }}>
            <span>classifier</span>
            <span>features</span>
            <span style={{ textAlign: 'right' }}>cv auc</span>
            <span style={{ textAlign: 'right' }}>val auc</span>
            <span style={{ textAlign: 'right' }}>test auc</span>
            <span style={{ textAlign: 'right' }}>apgr</span>
          </div>
          {runs.map((r) => {
            const chosen = isChosen(r.model, r.features)
            return (
              <div
                key={`${r.model}:${r.features}`}
                className="index__row index__row--hover"
                role="row"
                data-active={chosen}
                style={{ gridTemplateColumns: cols }}
              >
                <span style={{ fontWeight: chosen ? 500 : 400 }}>
                  {MODEL_LABEL[r.model]} {chosen && <span className="chip chip--strong">chosen</span>}
                </span>
                <span className="index__cell--muted" style={{ fontSize: 'var(--text-sm)' }}>
                  {FEATURE_LABEL[r.features]}
                </span>
                <span className="index__cell--num">
                  {num(r.cv_auc)} <span className="index__cell--muted">±{num(r.cv_auc_std, 2)}</span>
                </span>
                <span className="index__cell--num index__cell--muted">{num(r.val_auc)}</span>
                <span className="index__cell--num">{num(r.test_auc)}</span>
                <span className="index__cell--num">{num(r.test_apgr)}</span>
              </div>
            )
          })}
        </div>
      </div>
      <p className="meta" style={{ marginTop: 'var(--s-3)', lineHeight: 1.7 }}>
        the ± is the spread across cv folds. with ~45 queries per fold it is wide, so small gaps between rows are noise.
        {summary.mode === 'mock' &&
          ' in mock mode the "embed v4" vectors are a tf-idf + svd stand-in, so this column says little about real embed v4.'}
      </p>

      <div className="block">
        <h3 className="label" style={{ marginBottom: 'var(--s-3)' }}>
          the two models, per source
        </h3>
        <div style={{ overflowX: 'auto' }}>
          <div className="index" role="table" style={{ minWidth: 720 }}>
            <div className="index__head" role="row" style={{ gridTemplateColumns: srcCols }}>
              <span>source</span>
              <span style={{ textAlign: 'right' }}>n</span>
              <span style={{ textAlign: 'right' }}>r7b</span>
              <span style={{ textAlign: 'right' }}>cmd a</span>
              <span style={{ textAlign: 'right' }}>r7b ok</span>
              <span style={{ textAlign: 'right' }}>r7b / 1k</span>
              <span style={{ textAlign: 'right' }}>cmd a / 1k</span>
            </div>
            {[...sources, 'all' as const].map((s) => {
              const row = summary.per_source[s]
              return (
                <div key={s} className="index__row index__row--hover" role="row" style={{ gridTemplateColumns: srcCols }}>
                  <span>
                    <span style={{ fontWeight: s === 'all' ? 500 : 400 }}>{s === 'all' ? 'all queries' : SOURCE_LABEL[s]}</span>
                    {s !== 'all' && (
                      <span className="meta" style={{ display: 'block' }}>
                        {SOURCE_BLURB[s]}
                      </span>
                    )}
                  </span>
                  <span className="index__cell--num index__cell--muted">{row.n}</span>
                  <span className="index__cell--num">{pct(row.cheap.quality, 0)}</span>
                  <span className="index__cell--num">{pct(row.strong.quality, 0)}</span>
                  <span className="index__cell--num">{pct(row.r7b_good_enough, 0)}</span>
                  <span className="index__cell--num index__cell--muted">{perK(row.cheap.cost_per_1k_queries / 1000)}</span>
                  <span className="index__cell--num">{perK(row.strong.cost_per_1k_queries / 1000)}</span>
                </div>
              )
            })}
          </div>
        </div>
        <p className="meta" style={{ marginTop: 'var(--s-3)', lineHeight: 1.7 }}>
          quality is accuracy for ground-truth sources and the judge's 1–10 score rescaled to 0–1 for open-ended ones. "r7b
          ok" is the routing label: r7b was at least as right as command a (or, for open-ended, within one rubric point).
          costs are dollars per 1,000 queries from measured token counts.
        </p>
      </div>
    </Section>
  )
}

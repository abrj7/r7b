import { pct } from '@/lib/format'
import type { Summary } from '@/lib/types'
import { MockNotice } from './MockNotice'

export function Hero({ summary }: { summary: Summary }) {
  const h = summary.headline
  const ratio = summary.pricing.chat[summary.models.strong].output_per_m / summary.pricing.chat[summary.models.cheap].output_per_m
  const n = summary.split_sizes.test

  return (
    <header className="hero split enter">
      <div>
        <h1 className="hero__name">r7b</h1>
        <p className="hero__role">
          a learned model router
          <br />
          command r7b ↔ command a
          <br />
          embed v4 · scikit-learn
        </p>
      </div>
      <div className="prose" style={{ maxWidth: '64ch' }}>
        <p>
          <a className="link--cohere" href="https://docs.cohere.com/docs/models" target="_blank" rel="noopener noreferrer">
            cohere
          </a>
          's command a is the better model. it also costs about <strong>{Math.round(ratio)}×</strong> more per output
          token than command r7b, and plenty of queries don't need it.
        </p>
        <p>
          <strong>r7b</strong> is a classifier that reads a query (its embed v4 embedding plus a handful of cheap text
          statistics) and predicts whether r7b's answer will hold up against command a's. confident queries go to r7b;
          the rest escalate. below: how much that saves, what it costs in quality, and every query it routed.
        </p>

        <div className="stats">
          <div>
            <div className="stat__value num">{pct(h.test.cost_saved)}</div>
            <div className="stat__label">cost saved vs. always using command a</div>
          </div>
          <div>
            <div className="stat__value num">{pct(h.test.quality_retained)}</div>
            <div className="stat__label">of command a's quality kept</div>
          </div>
          <div>
            <div className="stat__value num">{pct(1 - h.test.frac_strong, 0)}</div>
            <div className="stat__label">of queries answered by r7b alone</div>
          </div>
        </div>
        <p className="meta" style={{ marginTop: 'var(--s-4)', fontSize: 'var(--text-xs)', lineHeight: 1.7 }}>
          {n} held-out test queries. the threshold was fixed on train + val to keep ≥{pct(h.target_retained, 0)} of
          command a's quality, before test was scored.
        </p>
        {summary.mode === 'mock' && <MockNotice />}
      </div>
    </header>
  )
}

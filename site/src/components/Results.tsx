import { useMemo, useState } from 'react'
import { Slider } from '@/components/ui/slider'
import { curveLabel, MODEL_COLOR, pct, SOURCE_LABEL } from '@/lib/format'
import { evaluateAt } from '@/lib/routing'
import type { Curves, QueryRow, Source, Summary } from '@/lib/types'
import { ParetoChart } from './ParetoChart'
import { Section } from './Section'

interface Props {
  summary: Summary
  curves: Curves
  queries: QueryRow[]
  threshold: number
  setThreshold: (t: number) => void
}

export function Results({ summary, curves, queries, threshold, setThreshold }: Props) {
  const test = useMemo(() => queries.filter((q) => q.split === 'test'), [queries])
  const [visible, setVisible] = useState<string[]>(() => Object.keys(curves.curves))
  const live = evaluateAt(test, threshold)
  const cheapCost = test.reduce((a, q) => a + q.r7b.cost, 0)
  const f = live.strongCost === cheapCost ? 0 : Math.min(1, Math.max(0, (live.cost - cheapCost) / (live.strongCost - cheapCost)))
  const randomQuality = (live.cheapQuality + f * (live.strongQuality - live.cheapQuality)) / live.strongQuality
  const lift = live.qualityRetained - randomQuality
  const b = summary.baselines

  const toggle = (k: string) =>
    setVisible((v) => (v.includes(k) ? v.filter((x) => x !== k) : [...v, k]))

  return (
    <Section
      id="results"
      label="results"
      note={`test split · ${test.length} queries never seen in training or threshold selection`}
      title="cost against quality"
      lede="every threshold on the router's confidence is a different trade: lower sends more to r7b and saves more, higher escalates more and keeps more quality. the dashed line is random routing: escalate a random share of queries. a router is only worth running where its curve sits above that line."
    >
      <div className="block">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 'var(--s-4)' }}>
          {Object.keys(curves.curves).map((k) => {
            const c = curves.curves[k]
            const isChosen = k === curves.chosen
            return (
              <button
                key={k}
                type="button"
                className="chip"
                aria-pressed={isChosen || visible.includes(k)}
                disabled={isChosen}
                onClick={() => toggle(k)}
                style={isChosen ? { cursor: 'default' } : undefined}
              >
                <span
                  aria-hidden
                  style={{
                    width: 10,
                    height: 2,
                    background: isChosen ? 'currentColor' : MODEL_COLOR[c.model],
                    display: 'inline-block',
                  }}
                />
                {curveLabel(k, curves)}
                {isChosen && ' · chosen'}
              </button>
            )
          })}
        </div>

        <ParetoChart
          curves={curves}
          visible={visible}
          live={{ cost: 1 - live.costSaved, quality: live.qualityRetained }}
          onPick={setThreshold}
        />

        <div className="panel" style={{ marginTop: 'var(--s-5)', padding: 'var(--s-5)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 'var(--s-4)', flexWrap: 'wrap' }}>
            <label htmlFor="threshold" className="label">
              escalate when confidence is below <span className="num" style={{ color: 'var(--ink)' }}>{threshold.toFixed(3)}</span>
            </label>
            <button type="button" className="textbtn" onClick={() => setThreshold(summary.headline.threshold)}>
              reset to the chosen threshold
            </button>
          </div>
          <Slider
            id="threshold"
            className="mt-4"
            min={0}
            max={1}
            step={0.001}
            value={[threshold]}
            onValueChange={([t]) => setThreshold(t)}
            aria-label="routing threshold"
          />
          <div className="stats" style={{ marginTop: 'var(--s-5)', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
            <Readout value={pct(live.costSaved)} label="cost saved" />
            <Readout value={pct(live.qualityRetained)} label="quality kept" />
            <Readout value={pct(live.fracStrong, 0)} label={`escalated (${live.n - live.toR7b} of ${live.n})`} />
            <Readout
              value={`${lift >= 0 ? '+' : ''}${(lift * 100).toFixed(1)}`}
              label="points of quality over random routing at the same cost"
            />
          </div>
          <p className="meta" style={{ marginTop: 'var(--s-4)', lineHeight: 1.7 }}>
            at this threshold the router sent {live.missedEscalations} queries to r7b that needed command a, and paid for
            command a on {live.wrongEscalations} that r7b would have handled.
          </p>
        </div>
      </div>

      <div className="block">
        <h3 className="label" style={{ marginBottom: 'var(--s-3)' }}>
          baselines on the same test queries
        </h3>
        <div className="index" role="table">
          <div className="index__head" role="row" style={{ gridTemplateColumns: 'minmax(0,1fr) 6rem 6rem 6rem' }}>
            <span>strategy</span>
            <span style={{ textAlign: 'right' }}>cost saved</span>
            <span style={{ textAlign: 'right' }}>quality</span>
            <span style={{ textAlign: 'right' }}>escalated</span>
          </div>
          {[
            ['always command a', b.always_command_a, 'the quality ceiling, at full price'],
            ['always r7b', b.always_r7b, 'the price floor'],
            [
              'random, same cost as router',
              { ...summary.headline.random_at_same_cost },
              'expected value of escalating a random share',
            ],
            ['r7b router (chosen threshold)', summary.headline.test, 'threshold fixed before test'],
            ['oracle', b.oracle, 'knows the label; the upper bound for any router'],
          ].map(([name, p, note]) => {
            const pt = p as Summary['baselines']['oracle']
            const isRouter = (name as string).startsWith('r7b router')
            return (
              <div
                key={name as string}
                className="index__row index__row--hover"
                role="row"
                style={{ gridTemplateColumns: 'minmax(0,1fr) 6rem 6rem 6rem' }}
              >
                <span>
                  <span style={{ fontWeight: isRouter ? 500 : 400 }}>{name as string}</span>
                  <span className="meta" style={{ display: 'block' }}>
                    {note as string}
                  </span>
                </span>
                <span className="index__cell--num">{pct(pt.cost_saved)}</span>
                <span className="index__cell--num">{pct(pt.quality_retained)}</span>
                <span className="index__cell--num">{pct(pt.frac_strong, 0)}</span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="block">
        <h3 className="label" style={{ marginBottom: 'var(--s-3)' }}>
          operating points, chosen on train + val, scored once on test
        </h3>
        <div className="index" role="table">
          <div className="index__head" role="row" style={{ gridTemplateColumns: '5rem 4.5rem minmax(0,1fr) minmax(0,1fr) 5.5rem' }}>
            <span>target</span>
            <span style={{ textAlign: 'right' }}>thresh.</span>
            <span style={{ textAlign: 'right' }}>cost saved (95% ci)</span>
            <span style={{ textAlign: 'right' }}>quality (95% ci)</span>
            <span style={{ textAlign: 'right' }}>random</span>
          </div>
          {summary.operating_points.map((op) => (
            <button
              type="button"
              key={op.target_retained}
              className="index__row index__row--hover index__row--button"
              role="row"
              onClick={() => setThreshold(op.threshold)}
              data-active={Math.abs(op.threshold - threshold) < 1e-9}
              style={{ gridTemplateColumns: '5rem 4.5rem minmax(0,1fr) minmax(0,1fr) 5.5rem' }}
              title="set the slider to this threshold"
            >
              <span className="mono" style={{ fontSize: 'var(--text-sm)' }}>
                ≥{pct(op.target_retained, 0)}
              </span>
              <span className="index__cell--num index__cell--muted">{op.threshold.toFixed(3)}</span>
              <span className="index__cell--num">
                {pct(op.test.cost_saved)}{' '}
                <span className="index__cell--muted">
                  [{pct(op.test_ci.cost_saved[0], 0)}, {pct(op.test_ci.cost_saved[1], 0)}]
                </span>
              </span>
              <span className="index__cell--num">
                {pct(op.test.quality_retained)}{' '}
                <span className="index__cell--muted">
                  [{pct(op.test_ci.quality_retained[0], 0)}, {pct(op.test_ci.quality_retained[1], 0)}]
                </span>
              </span>
              <span className="index__cell--num index__cell--muted">{pct(op.random_at_same_cost.quality_retained)}</span>
            </button>
          ))}
        </div>
        <p className="meta" style={{ marginTop: 'var(--s-3)', lineHeight: 1.7 }}>
          intervals are 1,000 bootstrap resamples of the test queries. "random" is the expected quality of random routing
          at the same cost. click a row to move the slider there.
        </p>
      </div>

      <div className="block">
        <h3 className="label" style={{ marginBottom: 'var(--s-3)' }}>
          where escalations go at this threshold
        </h3>
        <div style={{ display: 'grid', gap: 'var(--s-3)' }}>
          {(Object.keys(live.bySource) as Source[]).map((s) => {
            const { n, escalated } = live.bySource[s]
            const share = n ? escalated / n : 0
            return (
              <div key={s} style={{ display: 'grid', gridTemplateColumns: '7rem minmax(0,1fr) 5.5rem', gap: 'var(--s-4)', alignItems: 'center' }}>
                <span className="mono" style={{ fontSize: 'var(--text-sm)' }}>
                  {SOURCE_LABEL[s]}
                </span>
                <div className="meter" aria-hidden>
                  <div className="meter__fill" style={{ width: `${share * 100}%`, background: 'var(--rust)' }} />
                </div>
                <span className="index__cell--num index__cell--muted">
                  {escalated}/{n}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </Section>
  )
}

function Readout({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div className="stat__value num" style={{ fontSize: 'var(--text-xl)' }}>
        {value}
      </div>
      <div className="stat__label">{label}</div>
    </div>
  )
}

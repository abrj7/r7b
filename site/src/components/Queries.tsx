import { useMemo, useState } from 'react'
import { pct, SOURCE_LABEL, usd } from '@/lib/format'
import { verdict } from '@/lib/routing'
import type { QueryRow, Source, Split } from '@/lib/types'
import { Section } from './Section'

type Outcome = 'all' | 'r7b' | 'escalated' | 'missed' | 'overpaid'

const OUTCOMES: { id: Outcome; label: string }[] = [
  { id: 'all', label: 'all' },
  { id: 'r7b', label: 'kept on r7b' },
  { id: 'escalated', label: 'escalated' },
  { id: 'missed', label: 'should have escalated' },
  { id: 'overpaid', label: 'escalated needlessly' },
]

const PAGE = 20


export function Queries({ queries, threshold }: { queries: QueryRow[]; threshold: number }) {
  const [source, setSource] = useState<Source | 'all'>('all')
  const [split, setSplit] = useState<Split | 'all'>('test')
  const [outcome, setOutcome] = useState<Outcome>('all')
  const [search, setSearch] = useState('')
  const [open, setOpen] = useState<string | null>(null)
  const [limit, setLimit] = useState(PAGE)

  const rows = useMemo(() => {
    const s = search.trim().toLowerCase()
    return queries.filter((q) => {
      if (source !== 'all' && q.source !== source) return false
      if (split !== 'all' && q.split !== split) return false
      if (s && !q.text.toLowerCase().includes(s) && !q.id.includes(s)) return false
      const v = verdict(q, threshold)
      if (outcome === 'r7b' && !v.toR7b) return false
      if (outcome === 'escalated' && v.toR7b) return false
      if (outcome === 'missed' && v.tag !== 'missed') return false
      if (outcome === 'overpaid' && v.tag !== 'overpaid') return false
      return true
    })
  }, [queries, source, split, outcome, search, threshold])

  const reset = () => {
    setLimit(PAGE)
    setOpen(null)
  }

  return (
    <Section
      id="queries"
      label="queries"
      note="every query, both answers, and the call the router made"
      title="drill down"
      lede="confidence is the router's probability that r7b is good enough. for test queries it comes from the final router; for train and val queries it is a 5-fold out-of-fold prediction, so no query is scored by a model that saw it. decisions follow the threshold slider above."
    >
      <div style={{ display: 'grid', gap: 'var(--s-3)' }}>
        <input
          className="field"
          type="search"
          placeholder="search query text or id"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value)
            reset()
          }}
          aria-label="search queries"
        />
        <FilterRow
          label="source"
          value={source}
          options={[{ id: 'all', label: 'all' }, ...(['gsm8k', 'mmlu', 'nq', 'open'] as Source[]).map((s) => ({ id: s, label: SOURCE_LABEL[s] }))]}
          onChange={(v) => {
            setSource(v as Source | 'all')
            reset()
          }}
        />
        <FilterRow
          label="split"
          value={split}
          options={[
            { id: 'test', label: 'test' },
            { id: 'val', label: 'val' },
            { id: 'train', label: 'train' },
            { id: 'all', label: 'all' },
          ]}
          onChange={(v) => {
            setSplit(v as Split | 'all')
            reset()
          }}
        />
        <FilterRow
          label="outcome"
          value={outcome}
          options={OUTCOMES}
          onChange={(v) => {
            setOutcome(v as Outcome)
            reset()
          }}
        />
      </div>

      <div className="meta" style={{ margin: 'var(--s-5) 0 var(--s-2)' }}>
        {rows.length} {rows.length === 1 ? 'query' : 'queries'}
      </div>

      {rows.length === 0 ? (
        <div className="panel meta" style={{ padding: 'var(--s-5)', textAlign: 'center' }}>
          nothing matches these filters.{' '}
          <button
            type="button"
            className="textbtn"
            onClick={() => {
              setSource('all')
              setSplit('all')
              setOutcome('all')
              setSearch('')
              reset()
            }}
          >
            clear them
          </button>
        </div>
      ) : (
        <div className="index">
          {rows.slice(0, limit).map((q) => (
            <QueryItem
              key={q.id}
              q={q}
              threshold={threshold}
              open={open === q.id}
              onToggle={() => setOpen(open === q.id ? null : q.id)}
            />
          ))}
        </div>
      )}
      {rows.length > limit && (
        <div style={{ marginTop: 'var(--s-4)', textAlign: 'center' }}>
          <button type="button" className="textbtn" onClick={() => setLimit((l) => l + PAGE * 2)}>
            show more ({rows.length - limit} left)
          </button>
        </div>
      )}
    </Section>
  )
}

function FilterRow({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: { id: string; label: string }[]
  onChange: (v: string) => void
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6 }} role="group" aria-label={label}>
      <span className="label" style={{ width: '4.5rem' }}>
        {label}
      </span>
      {options.map((o) => (
        <button key={o.id} type="button" className="chip" aria-pressed={value === o.id} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

function QueryItem({ q, threshold, open, onToggle }: { q: QueryRow; threshold: number; open: boolean; onToggle: () => void }) {
  const v = verdict(q, threshold)
  return (
    <div className="index__row" style={{ display: 'block', padding: 0 }}>
      <button
        type="button"
        className="index__row index__row--hover index__row--button query-row"
        style={{ padding: 'var(--s-3) 0' }}
        onClick={onToggle}
        aria-expanded={open}
        data-active={open}
      >
        <span className="query-row__text">
          <span className="meta" style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--s-2)', alignItems: 'center' }}>
            {q.id}
            <span className="chip">{SOURCE_LABEL[q.source]}</span>
            {q.split !== 'test' && <span className="chip">{q.split}</span>}
          </span>
          <span
            style={{
              display: 'block',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: open ? 'normal' : 'nowrap',
              fontSize: 'var(--text-sm)',
              marginTop: 2,
            }}
          >
            {open ? '' : q.text.split('\n')[0]}
          </span>
        </span>
        <Confidence p={q.p} threshold={threshold} />
        <span className="query-row__verdict">
          <span className={`chip ${v.toR7b ? 'chip--r7b' : 'chip--a'}`}>{v.toR7b ? 'r7b' : 'command a'}</span>
          <span className={`chip ${v.tag === 'ok' ? 'chip--ok' : 'chip--warn'}`}>{v.text}</span>
        </span>
      </button>
      {open && <QueryDetail q={q} toR7b={v.toR7b} />}
    </div>
  )
}

export function Confidence({ p, threshold }: { p: number; threshold: number }) {
  return (
    <span style={{ display: 'grid', gap: 4 }} title={`confidence r7b is enough: ${pct(p)}`}>
      <span className="meta num" style={{ textAlign: 'right' }}>
        {pct(p, 0)}
      </span>
      <span className="meter" aria-hidden>
        <span className="meter__fill" style={{ width: `${p * 100}%`, background: p >= threshold ? 'var(--accent)' : 'var(--rust)' }} />
        <span className="meter__tick" style={{ left: `${threshold * 100}%` }} />
      </span>
    </span>
  )
}

function QueryDetail({ q, toR7b }: { q: QueryRow; toR7b: boolean }) {
  const gold = Array.isArray(q.gold) ? q.gold.join(' · ') : q.gold
  const chosenCost = (toR7b ? q.r7b.cost : q.command_a.cost) + q.router_cost
  return (
    <div className="enter" style={{ padding: '0 0 var(--s-5)' }}>
      <p className="answer" style={{ color: 'var(--ink)', fontSize: 'var(--text-base)', marginBottom: 'var(--s-3)' }}>
        {q.text}
      </p>
      <div className="meta" style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--s-4)', marginBottom: 'var(--s-4)' }}>
        {gold && <span>gold: {gold}</span>}
        {q.meta.subject && <span>subject: {q.meta.subject.replaceAll('_', ' ')}</span>}
        {q.meta.category && <span>category: {q.meta.category}</span>}
        <span>
          this route cost {usd(chosenCost)} vs {usd(q.command_a.cost)} for command a
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 'var(--s-4)' }}>
        <AnswerCard title="command r7b" a={q.r7b} picked={toR7b} source={q.source} />
        <AnswerCard title="command a" a={q.command_a} picked={!toR7b} source={q.source} />
      </div>
    </div>
  )
}

function AnswerCard({ title, a, picked, source }: { title: string; a: QueryRow['r7b']; picked: boolean; source: Source }) {
  const score = source === 'open' ? `judge ${a.judge_score}/10` : a.quality === 1 ? 'correct' : 'wrong'
  const good = source === 'open' ? (a.judge_score ?? 0) >= 7 : a.quality === 1
  return (
    <div className="panel" style={{ padding: 'var(--s-4)', borderColor: picked ? 'var(--ink-faint)' : undefined }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 'var(--s-2)', marginBottom: 'var(--s-2)' }}>
        <span className="mono" style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>
          {title}
        </span>
        <span style={{ display: 'flex', gap: 4 }}>
          {picked && <span className="chip chip--strong">routed here</span>}
          <span className={`chip ${good ? 'chip--ok' : 'chip--warn'}`}>{score}</span>
        </span>
      </div>
      <div className="answer" style={{ maxHeight: 260, overflowY: 'auto' }}>
        {a.text}
      </div>
      <div className="meta" style={{ marginTop: 'var(--s-3)' }}>
        {a.input_tokens} in · {a.output_tokens} out · {usd(a.cost)}
      </div>
    </div>
  )
}

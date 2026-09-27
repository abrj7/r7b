import { useMemo, useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { predict } from '@/lib/features'
import { SOURCE_LABEL, usd } from '@/lib/format'
import type { BrowserRouter, QueryRow } from '@/lib/types'
import { verdict } from '@/lib/routing'
import { Confidence } from './Queries'
import { Section } from './Section'

const EXAMPLES = [
  'What is the capital of Australia?',
  'Rewrite this to sound friendlier: "Send the file now."',
  'Prove that the square root of 2 is irrational, and explain each step.',
  'Write a Python function that returns the nth Fibonacci number in O(log n) time.',
  'A train leaves at 3:15pm going 60 mph. Another leaves the same station at 4:00pm going 80 mph. When does the second catch up?',
]

const FEATURE_HUMAN: Record<string, string> = {
  log_chars: 'length',
  log_words: 'word count',
  log_lines: 'line count',
  log_numbers: 'numbers in the text',
  choice_lines: 'multiple-choice options',
  question_marks: 'question marks',
  avg_word_len: 'average word length',
  capital_ratio: 'capitalized words',
  log_math_chars: 'math symbols',
  code_char_ratio: 'code punctuation',
  reasoning_words: 'reasoning verbs (explain, prove…)',
  constraint_words: 'constraints (exactly, only…)',
  simple_words: 'simple question words',
  starts_wh: 'starts with who/what/when…',
  log_sentences: 'sentence count',
  quotes: 'quoted text',
  has_currency: 'dollar amounts',
}

export function TryRouter({ queries, router, threshold }: { queries: QueryRow[]; router: BrowserRouter; threshold: number }) {
  return (
    <Section
      id="try"
      label="try it"
      note="runs in your browser on precomputed data; no api calls"
      title="try the router"
      lede="pick a query from the dataset to see the full router's call and what each model actually said, or type your own and get a verdict from a smaller router that only reads surface features (embedding new text would need an api key)."
    >
      <Tabs defaultValue="dataset">
        <TabsList variant="line" className="gap-4">
          <TabsTrigger value="dataset" className="mono px-0 text-[0.8125rem] tracking-[0.08em]">
            from the dataset
          </TabsTrigger>
          <TabsTrigger value="own" className="mono px-0 text-[0.8125rem] tracking-[0.08em]">
            your own query
          </TabsTrigger>
        </TabsList>
        <TabsContent value="dataset" className="mt-6">
          <FromDataset queries={queries} threshold={threshold} />
        </TabsContent>
        <TabsContent value="own" className="mt-6">
          <OwnQuery router={router} />
        </TabsContent>
      </Tabs>
    </Section>
  )
}

function FromDataset({ queries, threshold }: { queries: QueryRow[]; threshold: number }) {
  const test = useMemo(() => queries.filter((q) => q.split === 'test'), [queries])
  const [i, setI] = useState(0)
  const q = test[i % test.length]
  const v = verdict(q, threshold)
  const shuffle = () => {
    let j = i
    while (test.length > 1 && j === i) j = Math.floor(Math.random() * test.length)
    setI(j)
  }
  const picked = v.toR7b ? q.r7b : q.command_a
  const cost = picked.cost + q.router_cost

  return (
    <div className="panel" style={{ padding: 'var(--s-5)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 'var(--s-3)', alignItems: 'center' }}>
        <span className="meta">
          {q.id} · <span className="chip">{SOURCE_LABEL[q.source]}</span>
        </span>
        <span style={{ display: 'flex', gap: 'var(--s-4)' }}>
          <button type="button" className="textbtn" onClick={() => setI((i - 1 + test.length) % test.length)} aria-label="previous query">
            ← prev
          </button>
          <button type="button" className="textbtn" onClick={() => setI((i + 1) % test.length)} aria-label="next query">
            next →
          </button>
          <button type="button" className="textbtn" onClick={shuffle}>
            shuffle
          </button>
        </span>
      </div>
      <p className="answer" style={{ marginTop: 'var(--s-4)', color: 'var(--ink)', fontSize: 'var(--text-base)', maxHeight: 180, overflowY: 'auto' }}>
        {q.text}
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', gap: 'var(--s-5)', alignItems: 'end', marginTop: 'var(--s-5)' }}>
        <div>
          <div className="label" style={{ marginBottom: 'var(--s-2)' }}>
            confidence r7b is enough · threshold {threshold.toFixed(3)}
          </div>
          <Confidence p={q.p} threshold={threshold} />
        </div>
        <span className={`chip ${v.toR7b ? 'chip--r7b' : 'chip--a'}`} style={{ fontSize: 'var(--text-xs)' }}>
          → {v.toR7b ? 'command r7b' : 'command a'}
        </span>
      </div>

      <div className="stats" style={{ gridTemplateColumns: 'repeat(3, minmax(0,1fr))' }}>
        <div>
          <div className="stat__value" style={{ fontSize: 'var(--text-lg)' }}>
            {v.tag === 'ok' ? 'right call' : v.tag === 'missed' ? 'missed' : 'overpaid'}
          </div>
          <div className="stat__label">
            {v.tag === 'ok'
              ? v.toR7b
                ? 'r7b held up; command a was not needed'
                : 'command a was needed and got the query'
              : v.tag === 'missed'
                ? 'r7b fell short where command a did better'
                : 'r7b would have been as good'}
          </div>
        </div>
        <div>
          <div className="stat__value num" style={{ fontSize: 'var(--text-lg)' }}>
            {usd(cost)}
          </div>
          <div className="stat__label">this route, incl. the embed call</div>
        </div>
        <div>
          <div className="stat__value num" style={{ fontSize: 'var(--text-lg)' }}>
            {usd(q.command_a.cost)}
          </div>
          <div className="stat__label">always command a</div>
        </div>
      </div>

      <details style={{ marginTop: 'var(--s-5)' }}>
        <summary className="textbtn" style={{ display: 'inline-block' }}>
          show both answers
        </summary>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 'var(--s-4)', marginTop: 'var(--s-3)' }}>
          {(
            [
              ['command r7b', q.r7b],
              ['command a', q.command_a],
            ] as const
          ).map(([name, a]) => (
            <div key={name}>
              <div className="label" style={{ marginBottom: 4 }}>
                {name} · {q.source === 'open' ? `judge ${a.judge_score}/10` : a.quality === 1 ? 'correct' : 'wrong'}
              </div>
              <div className="answer" style={{ maxHeight: 220, overflowY: 'auto' }}>
                {a.text}
              </div>
            </div>
          ))}
        </div>
      </details>
    </div>
  )
}

function OwnQuery({ router }: { router: BrowserRouter }) {
  const [text, setText] = useState(EXAMPLES[2])
  const trimmed = text.trim()
  const { p, contributions } = useMemo(() => predict(router, trimmed), [router, trimmed])
  const toR7b = p >= router.threshold
  const top = [...contributions].sort((a, b) => Math.abs(b.push) - Math.abs(a.push)).slice(0, 5)
  const maxPush = Math.max(...top.map((c) => Math.abs(c.push)), 0.01)

  return (
    <div className="panel" style={{ padding: 'var(--s-5)' }}>
      <label htmlFor="own-query" className="label">
        your query
      </label>
      <textarea
        id="own-query"
        className="field"
        rows={3}
        style={{ marginTop: 'var(--s-2)', resize: 'vertical' }}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="ask anything"
      />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 'var(--s-3)' }}>
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" className="chip" aria-pressed={text === ex} onClick={() => setText(ex)}>
            {ex.length > 34 ? `${ex.slice(0, 32)}…` : ex}
          </button>
        ))}
      </div>

      {trimmed.length === 0 ? (
        <p className="meta" style={{ marginTop: 'var(--s-5)' }}>
          type something to route it.
        </p>
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', gap: 'var(--s-5)', alignItems: 'end', marginTop: 'var(--s-5)' }}>
            <div>
              <div className="label" style={{ marginBottom: 'var(--s-2)' }}>
                confidence r7b is enough · threshold {router.threshold.toFixed(3)}
              </div>
              <Confidence p={p} threshold={router.threshold} />
            </div>
            <span className={`chip ${toR7b ? 'chip--r7b' : 'chip--a'}`} style={{ fontSize: 'var(--text-xs)' }}>
              → {toR7b ? 'command r7b' : 'command a'}
            </span>
          </div>

          <div className="label" style={{ margin: 'var(--s-5) 0 var(--s-2)' }}>
            what moved the needle
          </div>
          <div style={{ display: 'grid', gap: 6 }}>
            {top.map((c) => (
              <div key={c.name} style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 8rem', gap: 'var(--s-3)', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--text-sm)', color: 'var(--ink-soft)' }}>{FEATURE_HUMAN[c.name] ?? c.name}</span>
                <span style={{ position: 'relative', height: 8 }} aria-label={`${c.push > 0 ? 'toward r7b' : 'toward command a'}`}>
                  <span style={{ position: 'absolute', left: '50%', top: -2, bottom: -2, width: 1, background: 'var(--line)' }} />
                  <span
                    style={{
                      position: 'absolute',
                      top: 1,
                      height: 6,
                      borderRadius: 2,
                      background: c.push > 0 ? 'var(--accent)' : 'var(--rust)',
                      width: `${(Math.abs(c.push) / maxPush) * 50}%`,
                      left: c.push > 0 ? '50%' : `${50 - (Math.abs(c.push) / maxPush) * 50}%`,
                      transition: 'all 0.3s var(--ease)',
                    }}
                  />
                </span>
              </div>
            ))}
          </div>
          <p className="meta" style={{ marginTop: 'var(--s-4)', lineHeight: 1.7 }}>
            blue pushes toward r7b, orange toward command a. this demo router is a logistic regression on 17 surface
            features (cv auc {router.cv_auc.toFixed(2)}, test auc {router.test_auc.toFixed(2)}); the full router adds a
            256-d embed v4 vector.
          </p>
        </>
      )}
    </div>
  )
}

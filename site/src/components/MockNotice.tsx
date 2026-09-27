import { REPO_URL } from './Nav'

export function MockNotice() {
  return (
    <div
      className="panel"
      role="note"
      style={{ marginTop: 'var(--s-5)', padding: 'var(--s-3) var(--s-4)', display: 'flex', gap: 'var(--s-3)', alignItems: 'baseline' }}
    >
      <span className="chip chip--warn">simulated</span>
      <span style={{ fontSize: 'var(--text-sm)', lineHeight: 1.6, color: 'var(--ink-soft)' }}>
        these results come from r7b's deterministic mock mode, which stands in for the cohere api until a key is
        added. the pipeline, splits, and metrics are real; the model answers are simulated.{' '}
        <a className="inline-link" href={`${REPO_URL}#running-it-for-real`} target="_blank" rel="noopener noreferrer">
          how to run it live
        </a>
      </span>
    </div>
  )
}

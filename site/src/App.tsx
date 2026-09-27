import { useEffect, useState } from 'react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { Hero } from './components/Hero'
import { Method } from './components/Method'
import { Models } from './components/Models'
import { Nav, REPO_URL } from './components/Nav'
import { Queries } from './components/Queries'
import { Results } from './components/Results'
import { TryRouter } from './components/TryRouter'
import { useSiteData } from './lib/data'
import type { SiteData } from './lib/types'

export default function App() {
  const [state, retry] = useSiteData()

  return (
    <TooltipProvider>
      <div className="shell">
        <Nav />
        {state.status === 'loading' && <Loading />}
        {state.status === 'error' && <LoadError message={state.error} onRetry={retry} />}
        {state.status === 'ready' && <Page data={state.data} />}
        <Footer generatedAt={state.status === 'ready' ? state.data.summary.generated_at : null} />
      </div>
    </TooltipProvider>
  )
}

function Page({ data }: { data: SiteData }) {
  const { summary, curves, queries, router } = data
  const [threshold, setThreshold] = useState(summary.headline.threshold)

  useEffect(() => {
    if (window.location.hash) document.getElementById(window.location.hash.slice(1))?.scrollIntoView()
  }, [])

  return (
    <main>
      <Hero summary={summary} />
      <Results summary={summary} curves={curves} queries={queries} threshold={threshold} setThreshold={setThreshold} />
      <Models summary={summary} />
      <Queries queries={queries} threshold={threshold} />
      <TryRouter queries={queries} router={router} threshold={threshold} />
      <Method summary={summary} />
    </main>
  )
}

function Loading() {
  return (
    <div className="hero split" aria-busy="true" aria-label="loading results">
      <div>
        <div className="skeleton" style={{ width: 80, height: 28 }} />
        <div className="skeleton" style={{ width: 150, height: 12, marginTop: 16 }} />
        <div className="skeleton" style={{ width: 120, height: 12, marginTop: 8 }} />
      </div>
      <div>
        <div className="skeleton" style={{ width: '90%', height: 16 }} />
        <div className="skeleton" style={{ width: '75%', height: 16, marginTop: 12 }} />
        <div className="skeleton" style={{ width: '100%', height: 320, marginTop: 48 }} />
      </div>
    </div>
  )
}

function LoadError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="hero split">
      <div className="label">error</div>
      <div className="panel" style={{ padding: 'var(--s-5)' }}>
        <p>the results data didn't load.</p>
        <p className="meta" style={{ marginTop: 'var(--s-2)' }}>
          {message}
        </p>
        <button type="button" className="textbtn" style={{ marginTop: 'var(--s-4)' }} onClick={onRetry}>
          try again
        </button>
      </div>
    </div>
  )
}

function Footer({ generatedAt }: { generatedAt: string | null }) {
  return (
    <footer className="footer meta">
      <span>
        built by{' '}
        <a className="inline-link" href="https://abdullahr.com">
          abdullah rajput
        </a>{' '}
        ·{' '}
        <a className="inline-link" href={REPO_URL} target="_blank" rel="noopener noreferrer">
          source on github
        </a>
      </span>
      {generatedAt && <span>results generated {generatedAt.slice(0, 10)}</span>}
    </footer>
  )
}

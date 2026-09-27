import { useEffect, useState } from 'react'
import type { BrowserRouter, Curves, QueryRow, SiteData, Summary } from './types'

async function getJson<T>(name: string): Promise<T> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${name}`)
  if (!res.ok) throw new Error(`${name}: ${res.status} ${res.statusText}`)
  return res.json() as Promise<T>
}

export type LoadState =
  | { status: 'loading' }
  | { status: 'error'; error: string }
  | { status: 'ready'; data: SiteData }

export function useSiteData(): [LoadState, () => void] {
  const [state, setState] = useState<LoadState>({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let cancelled = false
    Promise.all([
      getJson<Summary>('summary.json'),
      getJson<Curves>('curves.json'),
      getJson<QueryRow[]>('queries.json'),
      getJson<BrowserRouter>('browser_router.json'),
    ])
      .then(([summary, curves, queries, router]) => {
        if (!cancelled) setState({ status: 'ready', data: { summary, curves, queries, router } })
      })
      .catch((err: unknown) => {
        if (!cancelled) setState({ status: 'error', error: err instanceof Error ? err.message : String(err) })
      })
    return () => {
      cancelled = true
    }
  }, [attempt])

  return [
    state,
    () => {
      setState({ status: 'loading' })
      setAttempt((a) => a + 1)
    },
  ]
}

import { useEffect, useLayoutEffect, useRef, useState } from 'react'

const LINKS = [
  { id: 'results', label: 'results' },
  { id: 'queries', label: 'queries', optional: true },
  { id: 'try', label: 'try it' },
  { id: 'method', label: 'method' },
]

export const REPO_URL = 'https://github.com/abrj7/r7b'

export function Nav() {
  const [active, setActive] = useState<string | null>(null)
  const [stuck, setStuck] = useState(false)
  const [bar, setBar] = useState<{ left: number; width: number } | null>(null)
  const linksRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onScroll = () => setStuck(window.scrollY > 24)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    const sections = LINKS.map((l) => document.getElementById(l.id)).filter(Boolean) as HTMLElement[]
    const visible = new Map<string, number>()
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) visible.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0)
        const best = [...visible.entries()].filter(([, r]) => r > 0).sort((a, b) => b[1] - a[1])[0]
        setActive(best ? best[0] : null)
      },
      { rootMargin: '-20% 0px -55% 0px', threshold: [0, 0.1, 0.25, 0.5] },
    )
    sections.forEach((s) => io.observe(s))
    return () => io.disconnect()
  }, [])

  useLayoutEffect(() => {
    const root = linksRef.current
    if (!root || !active) {
      setBar(null)
      return
    }
    const el = root.querySelector<HTMLElement>(`[data-id="${active}"]`)
    if (!el || el.offsetParent === null) {
      setBar(null)
      return
    }
    setBar({ left: el.offsetLeft, width: el.offsetWidth })
  }, [active])

  return (
    <nav className="nav" data-stuck={stuck} aria-label="primary">
      <a className="nav__mark" href="https://abdullahr.com" aria-label="abdullah rajput, home">
        ar<span className="nav__mark-dot">.</span>
        <span className="nav__mark-sub"> / r7b</span>
      </a>
      <div className="nav__links" ref={linksRef}>
        {LINKS.map((l) => (
          <a
            key={l.id}
            data-id={l.id}
            href={`#${l.id}`}
            className={`nav__link${l.optional ? ' nav__link--optional' : ''}`}
            aria-current={active === l.id ? 'true' : undefined}
          >
            {l.label}
          </a>
        ))}
        <a className="nav__link" href={REPO_URL} target="_blank" rel="noopener noreferrer">
          code
        </a>
        <span
          className="nav__bar"
          aria-hidden
          style={{
            width: bar?.width ?? 0,
            transform: `translateX(${bar?.left ?? 0}px)`,
            opacity: bar ? 1 : 0,
          }}
        />
      </div>
    </nav>
  )
}

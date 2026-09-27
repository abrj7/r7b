import { useEffect, useRef, useState, type ReactNode } from 'react'

interface Props {
  id: string
  label: string
  note?: ReactNode
  title?: string
  lede?: ReactNode
  children: ReactNode
}

export function Section({ id, label, note, title, lede, children }: Props) {
  const ref = useRef<HTMLElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true)
          io.disconnect()
        }
      },
      { rootMargin: '0px 0px -10% 0px' },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  return (
    <section id={id} ref={ref} className="section split reveal" data-visible={visible} aria-labelledby={`${id}-title`}>
      <div className="split__label">
        {label}
        {note && <div className="split__note">{note}</div>}
      </div>
      <div style={{ minWidth: 0 }}>
        {title && (
          <h2 id={`${id}-title`} className="section__title">
            {title}
          </h2>
        )}
        {lede && <p className="section__lede">{lede}</p>}
        <div style={{ marginTop: title || lede ? 'var(--s-6)' : 0 }}>{children}</div>
      </div>
    </section>
  )
}

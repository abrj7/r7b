import { scaleLinear } from 'd3-scale'
import { line } from 'd3-shape'
import { useEffect, useMemo, useRef, useState } from 'react'
import { MODEL_COLOR, pct } from '@/lib/format'
import type { Curves, Point } from '@/lib/types'

interface Props {
  curves: Curves
  visible: string[]
  live: { cost: number; quality: number }
  onPick: (threshold: number) => void
}

const HEIGHT = 380
const M = { top: 16, right: 18, bottom: 44, left: 48 }

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(640)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

const xOf = (p: Point) => 1 - p.cost_saved
const yOf = (p: Point) => p.quality_retained

export function ParetoChart({ curves, visible, live, onPick }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<Point | null>(null)
  const { baselines } = curves
  const chosen = curves.curves[curves.chosen]

  const yMin = Math.max(0, Math.floor((baselines.always_r7b.quality_retained - 0.06) * 20) / 20)
  const x = useMemo(() => scaleLinear().domain([0, 1.02]).range([M.left, width - M.right]), [width])
  const y = useMemo(() => scaleLinear().domain([yMin, 1.03]).range([HEIGHT - M.bottom, M.top]), [yMin])
  const path = line<Point>()
    .x((p) => x(xOf(p)))
    .y((p) => y(yOf(p)))

  const byCost = (pts: Point[]) => [...pts].sort((a, b) => xOf(a) - xOf(b))
  const chosenPts = byCost(chosen.points)

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const rect = e.currentTarget.ownerSVGElement!.getBoundingClientRect()
    const cx = x.invert(e.clientX - rect.left)
    let best = chosenPts[0]
    for (const p of chosenPts) if (Math.abs(xOf(p) - cx) < Math.abs(xOf(best) - cx)) best = p
    setHover(best)
  }

  const xTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]
  const yTicks = y.ticks(6)
  const marker = (
    p: Point,
    label: string,
    shape: 'circle' | 'square' | 'diamond',
    anchor: 'start' | 'end' = 'start',
    dy = 3.5,
  ) => {
    const cx = x(xOf(p))
    const cy = y(yOf(p))
    const dx = anchor === 'start' ? 9 : -9
    return (
      <g key={label}>
        {shape === 'circle' && <circle cx={cx} cy={cy} r={4.5} fill="none" stroke="var(--ink)" strokeWidth={1.25} />}
        {shape === 'square' && <rect x={cx - 4} y={cy - 4} width={8} height={8} fill="var(--ink)" />}
        {shape === 'diamond' && (
          <path d={`M${cx},${cy - 5.5}L${cx + 5.5},${cy}L${cx},${cy + 5.5}L${cx - 5.5},${cy}Z`} fill="var(--blue)" />
        )}
        <text
          x={cx + dx}
          y={cy + dy}
          textAnchor={anchor}
          style={{ fill: 'var(--ink-soft)', paintOrder: 'stroke', stroke: 'rgba(232, 212, 198, 0.9)', strokeWidth: 4, strokeLinejoin: 'round' }}
        >
          {label}
        </text>
      </g>
    )
  }

  return (
    <div ref={ref} style={{ position: 'relative', width: '100%' }}>
      <svg
        className="plot"
        width={width}
        height={HEIGHT}
        role="img"
        aria-label="cost versus quality for each router, with baselines"
      >
        <g className="grid">
          {yTicks.map((t) => (
            <line key={t} x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} />
          ))}
        </g>
        <g className="axis">
          <line x1={M.left} x2={width - M.right} y1={HEIGHT - M.bottom} y2={HEIGHT - M.bottom} />
          {xTicks.map((t) => (
            <text key={t} x={x(t)} y={HEIGHT - M.bottom + 16} textAnchor="middle">
              {Math.round(t * 100)}%
            </text>
          ))}
          {yTicks.map((t) => (
            <text key={t} x={M.left - 8} y={y(t) + 3.5} textAnchor="end">
              {Math.round(t * 100)}%
            </text>
          ))}
          <text x={(M.left + width - M.right) / 2} y={HEIGHT - 8} textAnchor="middle">
            cost, as a share of always using command a →
          </text>
          <text transform={`translate(12 ${(M.top + HEIGHT - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
            quality, share of command a's →
          </text>
        </g>

        <path
          d={path(byCost(curves.random_line)) ?? ''}
          fill="none"
          stroke="var(--ink-faint)"
          strokeDasharray="3 4"
          strokeWidth={1}
        />

        {Object.entries(curves.curves)
          .filter(([k]) => visible.includes(k) && k !== curves.chosen)
          .map(([k, c]) => (
            <path
              key={k}
              className="curve"
              d={path(byCost(c.points)) ?? ''}
              stroke={MODEL_COLOR[c.model]}
              strokeDasharray={c.features === chosen.features ? undefined : '1 3'}
              opacity={0.75}
            />
          ))}
        <path className="curve curve--chosen" d={path(chosenPts) ?? ''} stroke="var(--ink)" />

        {marker(baselines.always_r7b, 'always r7b', 'circle', 'start', 17)}
        {marker(baselines.always_command_a, 'always command a', 'square', 'end', 17)}
        {marker(baselines.oracle, 'oracle', 'diamond')}

        <g style={{ transition: 'transform 0.3s var(--ease)' }} transform={`translate(${x(live.cost)} ${y(live.quality)})`}>
          <circle r={9} fill="rgba(31,27,22,0.1)" />
          <circle r={4.5} fill="var(--ink)" stroke="#fbeee6" strokeWidth={1.5} />
        </g>

        {hover && (
          <g pointerEvents="none">
            <line
              x1={x(xOf(hover))}
              x2={x(xOf(hover))}
              y1={M.top}
              y2={HEIGHT - M.bottom}
              stroke="var(--ink)"
              strokeOpacity={0.25}
            />
            <circle cx={x(xOf(hover))} cy={y(yOf(hover))} r={3.5} fill="none" stroke="var(--ink)" />
          </g>
        )}

        <rect
          x={M.left}
          y={M.top}
          width={Math.max(0, width - M.left - M.right)}
          height={HEIGHT - M.top - M.bottom}
          fill="transparent"
          style={{ cursor: 'crosshair' }}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          onClick={() => hover?.threshold != null && onPick(Math.min(1, hover.threshold))}
        />
      </svg>

      {hover && (
        <div
          className="panel meta"
          style={{
            position: 'absolute',
            pointerEvents: 'none',
            top: 8,
            left: Math.min(Math.max(x(xOf(hover)) + 12, M.left), width - 190),
            padding: '6px 9px',
            background: 'rgba(251,238,230,0.94)',
            lineHeight: 1.6,
            color: 'var(--ink-soft)',
          }}
        >
          threshold {hover.threshold != null ? Math.min(1, hover.threshold).toFixed(3) : '–'}
          <br />
          cost saved {pct(hover.cost_saved)} · quality {pct(hover.quality_retained)}
          <br />
          escalated {pct(hover.frac_strong, 0)} · click to set
        </div>
      )}
    </div>
  )
}

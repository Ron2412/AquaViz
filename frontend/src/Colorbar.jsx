import { cssColor } from './colormap.js'

// Vertical gradient legend that mirrors the mesh colormap, labeled with the
// current slice's value range.
export default function Colorbar({ min, max, units }) {
  if (min == null || max == null) return null
  const stops = Array.from({ length: 24 }, (_, i) => {
    const t = i / 23
    return `${cssColor(t)} ${(t * 100).toFixed(0)}%`
  }).join(', ')
  const mid = (min + max) / 2
  return (
    <div className="colorbar">
      <div className="colorbar-bar" style={{ background: `linear-gradient(to top, ${stops})` }} />
      <div className="colorbar-ticks">
        <span>{max.toFixed(1)}</span>
        <span>{mid.toFixed(1)}</span>
        <span>{min.toFixed(1)}</span>
      </div>
      <div className="colorbar-units">{units}</div>
    </div>
  )
}

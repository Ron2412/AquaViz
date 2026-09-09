// Perceptual-ish cool→warm ramp for scalar fields, input normalized to [0,1].
// Returns [r, g, b] in 0..1. Used for both the mesh vertex colors and the
// on-screen colorbar so they stay in sync.
const STOPS = [
  [0.0, [0.031, 0.318, 0.612]], // deep blue   (cold)
  [0.25, [0.129, 0.588, 0.753]], // teal
  [0.5, [0.400, 0.741, 0.647]], // green
  [0.75, [0.965, 0.749, 0.259]], // amber
  [1.0, [0.843, 0.188, 0.153]], // red         (warm)
]

export function colormap(t) {
  if (Number.isNaN(t)) return STOPS[0][1]
  t = Math.max(0, Math.min(1, t))
  for (let i = 1; i < STOPS.length; i++) {
    const [t1, c1] = STOPS[i]
    if (t <= t1) {
      const [t0, c0] = STOPS[i - 1]
      const f = (t - t0) / (t1 - t0 || 1)
      return [
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f,
      ]
    }
  }
  return STOPS[STOPS.length - 1][1]
}

/** CSS `rgb(...)` string for a normalized value — convenience for the colorbar. */
export function cssColor(t) {
  const [r, g, b] = colormap(t)
  return `rgb(${(r * 255) | 0}, ${(g * 255) | 0}, ${(b * 255) | 0})`
}

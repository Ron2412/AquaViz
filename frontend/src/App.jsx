import { useEffect, useRef, useState } from 'react'
import { getCoords, getTemperature } from './api.js'
import { OceanScene } from './OceanScene.js'
import Colorbar from './Colorbar.jsx'

export default function App() {
  const mountRef = useRef(null)
  const sceneRef = useRef(null)
  const [coords, setCoords] = useState(null)
  const [depthIndex, setDepthIndex] = useState(0)
  const [field, setField] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Create the Three.js scene once, tear it down on unmount.
  useEffect(() => {
    const scene = new OceanScene(mountRef.current)
    sceneRef.current = scene
    return () => {
      scene.dispose()
      sceneRef.current = null
    }
  }, [])

  // Fetch the coordinate catalogue once.
  useEffect(() => {
    getCoords()
      .then(setCoords)
      .catch((e) => setError(`Could not reach API: ${e.message}`))
  }, [])

  // Load the temperature slice whenever the selected depth changes.
  useEffect(() => {
    if (!coords) return
    const depth = coords.depths[depthIndex]
    setLoading(true)
    setError(null)
    getTemperature(depth)
      .then((f) => {
        setField(f)
        sceneRef.current?.setField(f)
      })
      .catch((e) => setError(`Slice request failed: ${e.message}`))
      .finally(() => setLoading(false))
  }, [coords, depthIndex])

  return (
    <div className="app">
      <div className="viewport" ref={mountRef} />
      <div className="panel">
        <h1>AquaViz</h1>
        <p className="sub">Temperature · {coords?.source ?? '…'} source</p>

        {error && <p className="error">{error}</p>}

        {coords && (
          <>
            <label className="control-label">
              Depth: <b>{coords.depths[depthIndex]} m</b>
            </label>
            <input
              type="range"
              min={0}
              max={coords.depths.length - 1}
              step={1}
              value={depthIndex}
              onChange={(e) => setDepthIndex(Number(e.target.value))}
            />
            <div className="hint">drag to orbit · scroll to zoom</div>
            {loading && <p className="loading">loading slice…</p>}
            {field && (
              <Colorbar
                min={field.value_min}
                max={field.value_max}
                units={field.units}
              />
            )}
          </>
        )}
      </div>
    </div>
  )
}

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { colormap } from './colormap.js'

// Imperative Three.js scene manager. React (App.jsx) owns the lifecycle: it
// constructs one OceanScene on mount, calls setField() whenever new grid data
// arrives, and dispose()s on unmount. Kept out of React's render loop so the
// WebGL context and geometry are created exactly once.
export class OceanScene {
  constructor(container) {
    this.container = container
    const w = container.clientWidth || 800
    const h = container.clientHeight || 600

    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x0b1622)

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 2000)
    this.camera.position.set(0, 70, 110)

    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.renderer.setSize(w, h)
    container.appendChild(this.renderer.domElement)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.75))
    const key = new THREE.DirectionalLight(0xffffff, 0.7)
    key.position.set(60, 120, 40)
    this.scene.add(key)

    this.mesh = null
    this.exaggeration = 8 // vertical relief applied to the normalized field

    this._onResize = () => this.resize()
    window.addEventListener('resize', this._onResize)

    this._animate = this._animate.bind(this)
    this._animate()
  }

  _animate() {
    this._raf = requestAnimationFrame(this._animate)
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  resize() {
    const w = this.container.clientWidth || 800
    const h = this.container.clientHeight || 600
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
  }

  /**
   * Build (or rebuild) the mesh from a FieldSlice payload:
   * { lat[], lon[], values[nlat][nlon], value_min, value_max }.
   * lon → x, lat → z, value → color (and a small y relief so it reads as 3D).
   */
  setField(field) {
    this._disposeMesh()
    const { lat, lon, values, value_min, value_max } = field
    const nlat = lat.length
    const nlon = lon.length
    const spanLon = lon[nlon - 1] - lon[0] || 1
    const spanLat = lat[nlat - 1] - lat[0] || 1
    const scale = 100 / Math.max(spanLon, spanLat)
    const midLon = lon[0] + spanLon / 2
    const midLat = lat[0] + spanLat / 2
    const range = value_max - value_min || 1

    const positions = new Float32Array(nlat * nlon * 3)
    const colors = new Float32Array(nlat * nlon * 3)
    let p = 0
    for (let j = 0; j < nlat; j++) {
      for (let i = 0; i < nlon; i++) {
        const v = values[j][i]
        const tn = v == null ? 0 : (v - value_min) / range
        positions[p] = (lon[i] - midLon) * scale
        positions[p + 1] = v == null ? 0 : tn * this.exaggeration
        positions[p + 2] = -(lat[j] - midLat) * scale
        const [r, g, b] = v == null ? [0.16, 0.16, 0.2] : colormap(tn)
        colors[p] = r
        colors[p + 1] = g
        colors[p + 2] = b
        p += 3
      }
    }

    const index = []
    for (let j = 0; j < nlat - 1; j++) {
      for (let i = 0; i < nlon - 1; i++) {
        const a = j * nlon + i
        const b = a + 1
        const c = a + nlon
        const d = c + 1
        index.push(a, c, b, b, c, d)
      }
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geo.setIndex(index)
    geo.computeVertexNormals()

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.85,
      metalness: 0.0,
    })
    this.mesh = new THREE.Mesh(geo, mat)
    this.scene.add(this.mesh)
  }

  _disposeMesh() {
    if (!this.mesh) return
    this.scene.remove(this.mesh)
    this.mesh.geometry.dispose()
    this.mesh.material.dispose()
    this.mesh = null
  }

  dispose() {
    cancelAnimationFrame(this._raf)
    window.removeEventListener('resize', this._onResize)
    this._disposeMesh()
    this.controls.dispose()
    this.renderer.dispose()
    const el = this.renderer.domElement
    if (el.parentNode) el.parentNode.removeChild(el)
  }
}

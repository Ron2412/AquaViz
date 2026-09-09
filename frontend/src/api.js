import axios from 'axios'

// Backend base URL. Override at build/run time with VITE_API_URL.
const baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({ baseURL, timeout: 15000 })

/** Coordinate catalogue: available variables, depth levels, timesteps. */
export async function getCoords() {
  const { data } = await api.get('/meta/coords')
  return data
}

/** One temperature slice (lat×lon grid) at a depth (m) and optional ISO time. */
export async function getTemperature(depth, time) {
  const params = { depth }
  if (time) params.time = time
  const { data } = await api.get('/temperature', { params })
  return data
}

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server runs on 5173 (matches the backend's CORS allow-list).
// Point the app at a non-default API with:  VITE_API_URL=http://host:port npm run dev
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
})

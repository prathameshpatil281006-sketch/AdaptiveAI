import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/monitor': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
  preview: {
    port: 5173,
  },
})

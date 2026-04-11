import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendPort = process.env.VITE_API_PORT || '8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: false,
    proxy: {
      '/api': `http://localhost:${backendPort}`,
    },
  },
})

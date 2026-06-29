import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base: './' keeps asset URLs relative so the bundle works when FastAPI
// serves it from '/'. The dev proxy lets `npm run dev` talk to the live
// backend on port 3000 (same contract as production same-origin).
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        ws: true,
      },
    },
  },
});

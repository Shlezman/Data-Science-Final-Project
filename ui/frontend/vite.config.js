import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const devSessionCookie = process.env.SENTISENSE_DEV_SESSION_COOKIE;
const devAuthHeaders = devSessionCookie ? { Cookie: devSessionCookie } : undefined;

// base: './' keeps asset URLs relative so the bundle works when FastAPI
// serves it from '/'. The dev proxy lets `npm run dev` talk to the live
// backend over its TLS endpoint (same contract as production same-origin).
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
        target: 'https://sentisens.cs.colman.ac.il',
        changeOrigin: true,
        headers: devAuthHeaders,
      },
      '/ws': {
        target: 'https://sentisens.cs.colman.ac.il',
        changeOrigin: true,
        headers: devAuthHeaders,
        ws: true,
      },
    },
  },
});

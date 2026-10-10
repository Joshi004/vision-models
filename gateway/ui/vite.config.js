import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Served by the model-api FastAPI process at /ui/ (see server.py's
// StaticFiles mount) -- `base` must match that prefix so the built
// index.html's asset URLs resolve correctly. The dev proxy below is only
// used for `npm run dev`; the production build talks to the API via
// same-origin fetches instead (see src/api.js).
//
// Tailwind v4's Vite plugin needs no separate tailwind.config.js -- the
// theme (see src/styles.css's `@theme` block) and source scanning are both
// handled from the single `@import "tailwindcss";` there.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/ui/',
  server: {
    proxy: {
      '/v1': 'http://localhost:8012',
    },
  },
});

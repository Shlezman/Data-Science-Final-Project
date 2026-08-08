import React from 'react';
import { createRoot } from 'react-dom/client';
// Rubik, self-hosted rather than pulled from a font CDN: the deployment host
// sits behind a firewall, and the typography should not depend on outbound
// network access. Each file carries every subset behind unicode-range, so the
// browser fetches only the Hebrew or Latin cut it actually needs.
import '@fontsource/rubik/400.css';
import '@fontsource/rubik/500.css';
import '@fontsource/rubik/600.css';
import '@fontsource/rubik/700.css';
import '@fontsource/rubik/800.css';
import App from './App.jsx';
import './styles.css';

function preferredTheme() {
  try {
    const stored = window.localStorage.getItem('sentisense-theme');
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // Storage can be unavailable in hardened/private browser contexts.
  }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

document.documentElement.dataset.theme = preferredTheme();

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

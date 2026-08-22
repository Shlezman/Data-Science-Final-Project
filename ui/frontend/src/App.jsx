import React, { useEffect, useState } from 'react';
import { getJson } from './lib/api.js';
import Dashboard from './components/Dashboard.jsx';
import Archive from './components/Archive.jsx';
import Simulator from './components/Simulator.jsx';
import Models from './components/Models.jsx';
import Infrastructure from './components/Infrastructure.jsx';
import Login from './components/Login.jsx';
import Centroids3D from './components/Centroids3D.jsx';

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'archive', label: 'Archive' },
  { id: 'simulator', label: 'Simulator' },
  { id: 'infrastructure', label: 'Infrastructure' },
];

/**
 * Root application shell. Owns the active-tab state (no router) and renders
 * the header with the current served-model version pulled once from /api/health.
 *
 * The Models panel is an OPERATOR view: it has no nav tab, but stays reachable
 * by clicking the "Serving: …" text in the header or by navigating to #models.
 *
 * The 3D-centroids drawer is opened from the header and mounted here rather than
 * inside the Dashboard, so it is reachable from every tab instead of floating over
 * the dashboard content in the bottom-right corner.
 *
 * @returns {JSX.Element} The full single-page app.
 */
export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [champion, setChampion] = useState(null);
  const [auth, setAuth] = useState(null);   // null = probing; {authed, gated, admin}
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || 'dark');
  const [centroidsOpen, setCentroidsOpen] = useState(false);

  const toggleTheme = () => {
    const nextTheme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = nextTheme;
    try {
      window.localStorage.setItem('sentisense-theme', nextTheme);
    } catch {
      // The visual switch should still work when storage is unavailable.
    }
    setTheme(nextTheme);
  };

  useEffect(() => {
    getJson('/api/auth')
      .then((res) => setAuth(res))
      .catch(() => setAuth({ authed: true, gated: false, admin: true }));   // gate endpoint down → don't lock out (dev)
  }, []);

  useEffect(() => {
    if (auth && !(auth.gated && !auth.authed)) {
      getJson('/api/health')
        .then((res) => setChampion(res?.champion ?? null))
        .catch(() => setChampion(null));
    }
  }, [auth]);

  useEffect(() => {
    if (!auth?.admin) return undefined;          // operator entrance is admin-only
    const onHash = () => {
      if (window.location.hash === '#models') setTab('models');
    };
    onHash();
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [auth]);

  if (auth === null) {
    return null;                       // brief blank while probing the gate
  }
  if (auth.gated && !auth.authed) {
    return (
      <Login
        onOk={(res) => setAuth({ authed: true, gated: true, admin: !!res?.admin })}
      />
    );
  }

  return (
    <div className="ss-app">
      <div className="ss-topbar">
        <header className="ss-header">
          <div className="ss-brand">
            {/* SentiSense mark, dark-surface variant: white swoosh, brand-blue
                dots. Drawn as vectors so it stays crisp at any size. The swoosh
                uses currentColor, so it follows the wordmark if the surface ever
                flips to light. The wordmark itself stays live text — searchable,
                screen-reader friendly, and sharp at every resolution. */}
            <svg className="ss-logo" viewBox="0 0 100 150" aria-hidden="true">
              <path d="M60 20 C80 28 80 56 55 76 C30 96 32 122 58 130"
                    fill="none" stroke="currentColor" strokeWidth="15" strokeLinecap="round" />
              <g fill="#7EB2E4">
                <circle cx="44" cy="11" r="6" />
                <circle cx="16" cy="17" r="7" />
                <circle cx="9" cy="45" r="6.5" />
                <circle cx="31" cy="67" r="6.5" />
                <circle cx="63" cy="79" r="6" />
                <circle cx="84" cy="105" r="6.5" />
                <circle cx="76" cy="137" r="6" />
              </g>
            </svg>
            <div>
              <h1 className="ss-title">
                <span className="ss-title__senti">Senti</span>Sense
              </h1>
              <p className="ss-tagline">Next-day TA-125 direction from Hebrew news sentiment</p>
            </div>
          </div>

          <div className="ss-header-actions">
            {/* Invisible operator entrance (opens Models) — admin-only, placed FIRST
                so it sits where the theme toggle used to be. */}
            {auth.admin ? (
              <span className="ss-champion ss-champion--ghost" onClick={() => setTab('models')}
                    role="button" tabIndex={-1} aria-hidden="true"
                    style={{ cursor: 'default', userSelect: 'none' }}>
                {champion ? (
                  <>
                    <span className="ss-champion__dot" aria-hidden="true" />
                    <span className="ss-champion__label">Serving</span>
                    <span className="ss-champion__value">{champion}</span>
                  </>
                ) : null}
              </span>
            ) : null}

            {/* Was a fixed pill floating over the dashboard's bottom-right corner,
                where it overlapped content and only existed on that one tab. */}
            <button
              type="button"
              className="ss-theme-toggle ss-centroids-toggle"
              onClick={() => setCentroidsOpen(true)}
              title="Daily news centroids (3D)"
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M12 2.6 20.5 7v10L12 21.4 3.5 17V7Z" />
                <path d="M3.5 7 12 11.6 20.5 7M12 11.6v9.8" />
              </svg>
              <span>3D centroids</span>
            </button>

            <button
              type="button"
              className="ss-theme-toggle"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M20.2 15.2A8.5 8.5 0 0 1 8.8 3.8 8.5 8.5 0 1 0 20.2 15.2Z" />
                </svg>
              )}
              <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
            </button>
          </div>
        </header>

        <nav className="ss-tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`ss-tab ${tab === t.id ? 'is-active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </div>

      <main>
        {tab === 'dashboard' ? <Dashboard /> : null}
        {tab === 'archive' ? <Archive /> : null}
        {tab === 'simulator' ? <Simulator /> : null}
        {tab === 'infrastructure' ? <Infrastructure /> : null}
        {tab === 'models' && auth.admin ? <Models /> : null}
      </main>

      <Centroids3D open={centroidsOpen} onClose={() => setCentroidsOpen(false)} />
    </div>
  );
}

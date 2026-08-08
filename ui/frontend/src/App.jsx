import React, { useEffect, useState } from 'react';
import { getJson } from './lib/api.js';
import Dashboard from './components/Dashboard.jsx';
import Archive from './components/Archive.jsx';
import Simulator from './components/Simulator.jsx';
import Models from './components/Models.jsx';

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'archive', label: 'Archive' },
  { id: 'simulator', label: 'Simulator' },
];

/**
 * Root application shell. Owns the active-tab state (no router) and renders
 * the header with the current served-model version pulled once from /api/health.
 *
 * The Models panel is an OPERATOR view: it has no nav tab, but stays reachable
 * by clicking the "Serving: …" text in the header or by navigating to #models.
 *
 * @returns {JSX.Element} The full single-page app.
 */
export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [champion, setChampion] = useState(null);

  useEffect(() => {
    getJson('/api/health')
      .then((res) => setChampion(res?.champion ?? null))
      .catch(() => setChampion(null));
  }, []);

  useEffect(() => {
    const onHash = () => {
      if (window.location.hash === '#models') setTab('models');
    };
    onHash();
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

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

          <span className="ss-champion" onClick={() => setTab('models')}
                role="button" tabIndex={-1}
                style={{ cursor: 'default', userSelect: 'none' }}>
            {champion ? (
              <>
                <span className="ss-champion__dot" aria-hidden="true" />
                <span className="ss-champion__label">Serving</span>
                <span className="ss-champion__value">{champion}</span>
              </>
            ) : null}
          </span>
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
        {tab === 'models' ? <Models /> : null}
      </main>
    </div>
  );
}

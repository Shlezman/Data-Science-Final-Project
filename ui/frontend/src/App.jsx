import React, { useEffect, useState } from 'react';
import { getJson } from './lib/api.js';
import Dashboard from './components/Dashboard.jsx';
import Archive from './components/Archive.jsx';
import Simulator from './components/Simulator.jsx';

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'archive', label: 'Archive' },
  { id: 'simulator', label: 'Simulator' },
];

/**
 * Root application shell. Owns the active-tab state (no router) and renders
 * the header with the current champion version pulled once from /api/health.
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

  return (
    <div className="ss-app">
      <header className="ss-header">
        <h1 className="ss-title">SentiSense</h1>
        <span className="ss-champion">
          {champion ? `Champion: ${champion}` : ''}
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

      <main>
        {tab === 'dashboard' ? <Dashboard /> : null}
        {tab === 'archive' ? <Archive /> : null}
        {tab === 'simulator' ? <Simulator /> : null}
      </main>
    </div>
  );
}

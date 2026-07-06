import React, { useEffect, useState, useCallback } from 'react';
import { getJson } from '../lib/api.js';
import { pct, direction, outcome } from '../lib/format.js';
import HeadlineList from './HeadlineList.jsx';
import Hero from './Hero.jsx';
import EdaPanels from './EdaPanels.jsx';
import Centroids3D from './Centroids3D.jsx';

const REFRESH_MS = 60_000;

/**
 * Renders the last-run banner derived from /api/health.
 *
 * @param {object} props Component props.
 * @param {object|null} props.lastRun The health payload's `last_run` object.
 * @returns {JSX.Element|null} The banner, or null if no data yet.
 */
function LastRunBanner({ lastRun }) {
  if (!lastRun) {
    return null;
  }
  const variant = lastRun.error
    ? 'is-error'
    : lastRun.skipped
      ? 'is-skip'
      : '';
  return (
    <div className={`ss-banner ${variant}`}>
      <span>
        <b>Today:</b> {lastRun.today || '—'}
      </span>
      <span>
        <b>Last success:</b> {lastRun.last_success || '—'}
      </span>
      {lastRun.prediction !== undefined && lastRun.prediction !== null ? (
        <span>
          <b>Prediction:</b> {String(lastRun.prediction)}
        </span>
      ) : null}
      {lastRun.skipped ? <span>Run skipped</span> : null}
      {lastRun.error ? (
        <span className="ss-error-text">Error: {lastRun.error}</span>
      ) : null}
    </div>
  );
}

/**
 * Renders a single labeled statistic card, optionally with a secondary line
 * (e.g. the model's held-out evaluation score next to the live value).
 *
 * @param {object} props Component props.
 * @param {string} props.label The stat name.
 * @param {string|number} props.value The stat value.
 * @param {string} [props.sub] Optional secondary line under the value.
 * @returns {JSX.Element} The stat card.
 */
function Stat({ label, value, sub }) {
  return (
    <div className="ss-stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub ? <div className="label" style={{ marginTop: 2 }}>{sub}</div> : null}
    </div>
  );
}

/**
 * Formats a metric that may be null into a fixed-precision string.
 *
 * @param {number|null|undefined} value The metric value.
 * @returns {string} Formatted value or "—".
 */
function metric(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(3);
}

/**
 * Dashboard view: served-model hero + last-run banner, metric stat cards,
 * recent predictions table, and the last-day live headlines list. Polls
 * /api/dashboard and /api/health every 60 seconds.
 *
 * @returns {JSX.Element} The dashboard.
 */
export default function Dashboard() {
  const [dashboard, setDashboard] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const [d, h] = await Promise.all([
        getJson('/api/dashboard'),
        getJson('/api/health'),
      ]);
      setDashboard(d);
      setHealth(h);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);

  if (error && !dashboard) {
    return <p className="ss-error-text">Failed to load dashboard: {error}</p>;
  }
  if (!dashboard) {
    return <p className="ss-muted">Loading dashboard…</p>;
  }

  const c = dashboard.confusion || {};
  const ev = dashboard.eval_metrics || null;
  const combined = dashboard.combined || null;
  const liveN = combined?.n_live ?? 0;
  const recent = dashboard.recent || [];
  const latest = dashboard.latest_headlines || {};

  return (
    <div>
      <Hero />
      <LastRunBanner lastRun={health?.last_run} />

      <div className="ss-card">
        <h2>
          Model performance
          {dashboard.model_type ? <span className="ss-tag">{dashboard.model_type}</span> : null}
        </h2>
        <p className="ss-muted">Serving: {dashboard.champion || '—'}</p>
        <p className="ss-section-title">
          Metrics <span className="ss-tag">Overall (eval + live)</span>
        </p>
        <div className="ss-stat-grid">
          <Stat label="Accuracy"
                value={combined ? metric(combined.accuracy) : metric(c.accuracy)}
                sub={combined
                  ? `eval ${metric(ev.accuracy)} + ${combined.n_live} live day${combined.n_live === 1 ? '' : 's'}`
                  : (ev?.accuracy != null ? `eval ${metric(ev.accuracy)}` : null)} />
          <Stat label="MCC"
                value={liveN > 0 ? metric(c.mcc) : (ev?.mcc != null ? metric(ev.mcc) : '—')}
                sub={liveN > 0 && ev?.mcc != null ? `eval ${metric(ev.mcc)}` : (liveN === 0 ? 'eval' : null)} />
          <Stat label="ROC-AUC" value={ev?.roc_auc != null ? metric(ev.roc_auc) : '—'}
                sub={ev?.roc_auc != null ? 'eval' : null} />
          <Stat label="Precision" value={liveN > 0 ? metric(c.precision) : '—'}
                sub={liveN > 0 ? 'live' : 'live — pending first settled day'} />
          <Stat label="Recall" value={liveN > 0 ? metric(c.recall) : '—'}
                sub={liveN > 0 ? 'live' : 'live — pending first settled day'} />
          <Stat label="F1" value={liveN > 0 ? metric(c.f1) : '—'}
                sub={liveN > 0 ? 'live' : 'live — pending first settled day'} />
          <Stat label="N" value={combined ? combined.n : (c.n ?? 0)}
                sub={combined ? `${combined.n_eval} eval + ${combined.n_live} live` : null} />
          <Stat label="Pending" value={c.pending ?? 0} />
        </div>
      </div>

      <EdaPanels />
      <Centroids3D />

      <div className="ss-card">
        <h2>Recent predictions</h2>
        {recent.length === 0 ? (
          <p className="ss-muted">No predictions yet.</p>
        ) : (
          <table className="ss-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Predicted</th>
                <th>Confidence</th>
                <th>Actual</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((r) => (
                <tr key={r.date}>
                  <td>{r.date}</td>
                  <td>{direction(r.prediction)}</td>
                  <td>{pct(r.confidence)}</td>
                  <td>{direction(r.actual)}</td>
                  <td>{outcome(r.prediction, r.actual)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="ss-card">
        <h2>
          Live headlines (last day)
          {latest.date ? <span className="ss-tag">{latest.date}</span> : null}
          {typeof latest.total === 'number' ? (
            <span className="ss-tag">{latest.total} total</span>
          ) : null}
        </h2>
        <HeadlineList headlines={latest.headlines} />
      </div>
    </div>
  );
}

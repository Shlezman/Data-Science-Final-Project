import React, { useEffect, useState, useCallback } from 'react';
import { getJson } from '../lib/api.js';
import { pct, direction, outcome } from '../lib/format.js';
import HeadlineList from './HeadlineList.jsx';

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
 * Renders a single labeled statistic card.
 *
 * @param {object} props Component props.
 * @param {string} props.label The stat name.
 * @param {string|number} props.value The stat value.
 * @returns {JSX.Element} The stat card.
 */
function Stat({ label, value }) {
  return (
    <div className="ss-stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
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
 * Renders the 2x2 confusion matrix from the dashboard `confusion` object.
 *
 * @param {object} props Component props.
 * @param {object} props.c The confusion counts (tp/tn/fp/fn).
 * @returns {JSX.Element} The matrix grid.
 */
function ConfusionMatrix({ c }) {
  return (
    <div className="ss-confusion">
      <span className="corner" />
      <span className="axis">Actual up</span>
      <span className="axis">Actual down</span>

      <span className="axis">Pred up</span>
      <div className="ss-cell tp">
        <div className="cell-label">TP</div>
        <div className="cell-value">{c.tp ?? 0}</div>
      </div>
      <div className="ss-cell fp">
        <div className="cell-label">FP</div>
        <div className="cell-value">{c.fp ?? 0}</div>
      </div>

      <span className="axis">Pred down</span>
      <div className="ss-cell fn">
        <div className="cell-label">FN</div>
        <div className="cell-value">{c.fn ?? 0}</div>
      </div>
      <div className="ss-cell tn">
        <div className="cell-label">TN</div>
        <div className="cell-value">{c.tn ?? 0}</div>
      </div>
    </div>
  );
}

/**
 * Dashboard view: champion + last-run banner, confusion matrix, stat cards,
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
  const recent = dashboard.recent || [];
  const latest = dashboard.latest_headlines || {};

  return (
    <div>
      <LastRunBanner lastRun={health?.last_run} />

      <div className="ss-card">
        <h2>Model performance</h2>
        <p className="ss-muted">Champion: {dashboard.champion || '—'}</p>
        <div className="ss-graph-wrap">
          <div style={{ flex: '0 0 auto' }}>
            <p className="ss-section-title">Confusion matrix</p>
            <ConfusionMatrix c={c} />
          </div>
          <div style={{ flex: '1 1 360px' }}>
            <p className="ss-section-title">Metrics</p>
            <div className="ss-stat-grid">
              <Stat label="Accuracy" value={metric(c.accuracy)} />
              <Stat label="Precision" value={metric(c.precision)} />
              <Stat label="Recall" value={metric(c.recall)} />
              <Stat label="F1" value={metric(c.f1)} />
              <Stat label="MCC" value={metric(c.mcc)} />
              <Stat label="N" value={c.n ?? 0} />
              <Stat label="Pending" value={c.pending ?? 0} />
            </div>
          </div>
        </div>
      </div>

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

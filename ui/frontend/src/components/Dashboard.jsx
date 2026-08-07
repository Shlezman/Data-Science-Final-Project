import React, { useEffect, useState, useCallback } from 'react';
import { getJson } from '../lib/api.js';
import { pct, direction, directionCls, outcome, outcomeCls, toneFromChance } from '../lib/format.js';
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
 * A small radial progress ring for a 0..1 fraction, drawn as plain SVG (no
 * charting lib needed for one gauge). Purely decorative — `aria-hidden`,
 * since the same value is already shown as text next to it.
 *
 * @param {object} props Component props.
 * @param {number|null|undefined} props.value Fraction in [0, 1].
 * @param {'pos'|'neg'|null} [props.tone] Ring color: pos=green, neg=red,
 *   else the app accent blue.
 * @returns {JSX.Element} The gauge SVG.
 */
function Gauge({ value, tone }) {
  const frac = typeof value === 'number' && !Number.isNaN(value)
    ? Math.max(0, Math.min(1, value)) : 0;
  const r = 44;
  const c = 2 * Math.PI * r;
  const color = tone === 'pos' ? 'var(--ss-pos)' : tone === 'neg' ? 'var(--ss-neg)' : 'var(--ss-accent)';
  return (
    <svg width="110" height="110" viewBox="0 0 110 110" className="ss-gauge" aria-hidden="true">
      <circle cx="55" cy="55" r={r} fill="none" stroke="var(--ss-border)" strokeWidth="9" />
      <circle cx="55" cy="55" r={r} fill="none" stroke={color} strokeWidth="9"
              strokeLinecap="round" strokeDasharray={c}
              strokeDashoffset={c * (1 - frac)} transform="rotate(-90 55 55)" />
      <text x="55" y="61" textAnchor="middle" fontSize="22" fontWeight="700" fill={color}>
        {Math.round(frac * 100)}%
      </text>
    </svg>
  );
}

/**
 * A full-width stacked bar showing how a total splits into parts, with an
 * inline legend. Used on the wide N tile so its extra width carries the
 * eval/live composition instead of sitting empty.
 *
 * @param {object} props Component props.
 * @param {Array<{label: string, value: number, color: string}>} props.parts
 *   Segments, rendered left to right in the given order.
 * @returns {JSX.Element|null} The bar, or null when the total is zero.
 */
function SplitBar({ parts }) {
  const total = parts.reduce((sum, p) => sum + (p.value || 0), 0);
  if (!total) {
    return null;
  }
  return (
    <div className="ss-splitbar">
      <div className="ss-splitbar__track" aria-hidden="true">
        {parts.map((p) => (
          <span key={p.label} className="ss-splitbar__seg"
                style={{ width: `${(p.value / total) * 100}%`, background: p.color }} />
        ))}
      </div>
      <div className="ss-splitbar__legend">
        {parts.map((p) => (
          <span key={p.label}>
            <span className="ss-splitbar__swatch" style={{ background: p.color }} aria-hidden="true" />
            {p.value} {p.label}
          </span>
        ))}
      </div>
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
 * @param {'pos'|'neg'|null} [props.tone] Optional good/weak visual cue,
 *   e.g. from {@link toneFromChance}. Reserved for metrics with a real
 *   chance baseline (accuracy, MCC, ROC-AUC) — carries meaning.
 * @param {'teal'|'violet'|'amber'|'blue'|'slate'} [props.accent] Purely
 *   decorative color swatch for metrics with no good/bad baseline
 *   (precision/recall/F1/N/pending), so the grid isn't all-grey without
 *   implying a judgement `tone` doesn't back up.
 * @param {string} [props.area] Bento grid-area name (see `.ss-stat-grid--metrics`).
 * @param {boolean} [props.featured] Renders as the large anchor tile.
 * @param {JSX.Element} [props.extra] Optional decoration (e.g. a {@link Gauge})
 *   rendered beside the text, only meaningful on featured tiles.
 * @param {JSX.Element} [props.below] Optional full-width element under the
 *   text (e.g. a {@link SplitBar}), for wide tiles that would otherwise
 *   leave most of their width empty.
 * @returns {JSX.Element} The stat card.
 */
function Stat({ label, value, sub, tone, accent, area, featured, extra, below }) {
  const cls = ['ss-stat'];
  if (tone) cls.push(`ss-stat--${tone}`);
  if (accent) cls.push(`ss-stat--accent-${accent}`);
  if (featured) cls.push('ss-stat--featured');
  if (area) cls.push(`ss-stat--area-${area}`);
  return (
    <div className={cls.join(' ')}>
      <div className="ss-stat__body">
        <div className="label">
          <span className="ss-stat__dot" aria-hidden="true" />
          {label}
        </div>
        <div className="value">{value}</div>
        {sub ? <div className="label" style={{ marginTop: 2 }}>{sub}</div> : null}
        {below || null}
      </div>
      {extra || null}
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
  const accValue = combined ? combined.accuracy : c.accuracy;
  const accTone = toneFromChance(accValue, 0.5);

  return (
    <div>
      <Hero />
      <LastRunBanner lastRun={health?.last_run} />

      <div className="ss-card">
        <h2>
          Model performance
          {dashboard.model_type ? <span className="ss-tag">{dashboard.model_type}</span> : null}
        </h2>
        <p className="ss-section-title">
          Metrics <span className="ss-tag">Overall (eval + live)</span>
        </p>
        <div className="ss-stat-grid ss-stat-grid--metrics">
          <Stat label="Accuracy" area="acc" featured
                value={metric(accValue)}
                sub={combined
                  ? `eval ${metric(ev.accuracy)} + ${combined.n_live} live day${combined.n_live === 1 ? '' : 's'}`
                  : (ev?.accuracy != null ? `eval ${metric(ev.accuracy)}` : null)}
                tone={accTone}
                extra={<Gauge value={accValue} tone={accTone} />} />
          <Stat label="MCC" area="mcc"
                value={liveN > 0 ? metric(c.mcc) : (ev?.mcc != null ? metric(ev.mcc) : '—')}
                sub={liveN > 0 && ev?.mcc != null ? `eval ${metric(ev.mcc)}` : (liveN === 0 ? 'eval' : null)}
                tone={toneFromChance(liveN > 0 ? c.mcc : ev?.mcc, 0)} />
          <Stat label="ROC-AUC" area="auc" value={ev?.roc_auc != null ? metric(ev.roc_auc) : '—'}
                sub={ev?.roc_auc != null ? 'eval' : null}
                tone={toneFromChance(ev?.roc_auc, 0.5)} />
          <Stat label="Precision" area="prec" accent="teal" value={liveN > 0 ? metric(c.precision) : '—'}
                sub={liveN > 0 ? 'live' : 'live — pending first settled day'} />
          <Stat label="Recall" area="rec" accent="violet" value={liveN > 0 ? metric(c.recall) : '—'}
                sub={liveN > 0 ? 'live' : 'live — pending first settled day'} />
          <Stat label="F1" area="f1" accent="amber" value={liveN > 0 ? metric(c.f1) : '—'}
                sub={liveN > 0 ? 'live' : 'live — pending first settled day'} />
          <Stat label="N" area="n" accent="blue" value={combined ? combined.n : (c.n ?? 0)}
                below={combined ? (
                  <SplitBar parts={[
                    { label: 'eval', value: combined.n_eval, color: '#60a5fa' },
                    { label: 'live', value: combined.n_live, color: '#fbbf24' },
                  ]} />
                ) : null} />
          <Stat label="Pending" area="pend" accent="slate" value={c.pending ?? 0} />
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
              {recent.map((r) => {
                const result = outcome(r.prediction, r.actual);
                return (
                  <tr key={r.date}>
                    <td>{r.date}</td>
                    <td>
                      <span className={`ss-badge ${directionCls(r.prediction)}`}>
                        {direction(r.prediction)}
                      </span>
                    </td>
                    <td>{pct(r.confidence)}</td>
                    <td>
                      <span className={`ss-badge ${directionCls(r.actual)}`}>
                        {direction(r.actual)}
                      </span>
                    </td>
                    <td>
                      <span className={`ss-badge ${outcomeCls(result)}`}>{result}</span>
                    </td>
                  </tr>
                );
              })}
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

import React, { useEffect, useState, useCallback } from 'react';
import { getJson } from '../lib/api.js';
import { pct, direction, directionCls, outcome, outcomeCls } from '../lib/format.js';
import HeadlineList from './HeadlineList.jsx';
import Hero from './Hero.jsx';
import EdaPanels from './EdaPanels.jsx';
import Centroids3D from './Centroids3D.jsx';

const REFRESH_MS = 60_000;

/**
 * Renders the last-run banner derived from /api/health, but ONLY when the run
 * needs attention (errored or was skipped). A healthy run's timestamp is shown
 * quietly in the {@link Hero} metadata instead — a full-width banner repeating
 * routine status added noise, and when the orchestrator had written no status
 * at all it rendered as a shell of em-dashes that read as broken. Note `{}` is
 * truthy, so the emptiness check has to look at the keys.
 *
 * @param {object} props Component props.
 * @param {object|null} props.lastRun The health payload's `last_run` object.
 * @returns {JSX.Element|null} The banner, or null when there is nothing wrong.
 */
function LastRunBanner({ lastRun }) {
  if (!lastRun || Object.keys(lastRun).length === 0) {
    return null;
  }
  if (!lastRun.error && !lastRun.skipped) {
    return null;
  }
  return (
    <div className={`ss-banner ${lastRun.error ? 'is-error' : 'is-skip'}`}>
      {lastRun.today ? (
        <span>
          <b>Today:</b> {lastRun.today}
        </span>
      ) : null}
      {lastRun.last_success ? (
        <span>
          <b>Last success:</b> {lastRun.last_success}
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

function metricPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

function metricAssessment(kind, value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return { label: 'No data', tone: 'neutral' };
  }
  if (kind === 'accuracy') {
    const delta = (value - 0.5) * 100;
    return { label: `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} pp vs baseline`, tone: 'neutral' };
  }
  if (kind === 'auc') {
    const delta = value - 0.5;
    return { label: `${delta >= 0 ? '+' : ''}${delta.toFixed(3)} vs baseline`, tone: 'neutral' };
  }
  return { label: `${value >= 0 ? '+' : ''}${value.toFixed(3)} vs baseline`, tone: 'neutral' };
}

function MetricInfo({ text }) {
  return (
    <span className="ss-metric-info" role="img" tabIndex="0" aria-label={text} title={text}>i</span>
  );
}

function CoreMetric({ label, value, displayValue, kind, baseline, domain, scope, info, comparison }) {
  const assessment = metricAssessment(kind, value);
  const [min, max] = domain;
  const clamp = (number) => Math.max(0, Math.min(100, ((number - min) / (max - min)) * 100));
  const valuePosition = typeof value === 'number' ? clamp(value) : clamp(baseline);
  const baselinePosition = clamp(baseline);
  const start = Math.min(valuePosition, baselinePosition);
  const width = Math.abs(valuePosition - baselinePosition);

  return (
    <article className={`ss-core-metric is-${assessment.tone}`}>
      <div className="ss-core-metric__head">
        <span>{label} <MetricInfo text={info} /></span>
        <span className="ss-metric-scope">{scope}</span>
      </div>
      <strong className="ss-core-metric__value">{displayValue}</strong>
      <span className="ss-core-metric__assessment">{assessment.label}</span>
      <div className="ss-baseline-scale" aria-hidden="true">
        <span className="ss-baseline-scale__delta" style={{ left: `${start}%`, width: `${Math.max(width, 0.6)}%` }} />
        <span className="ss-baseline-scale__baseline" style={{ left: `${baselinePosition}%` }} />
        <span className="ss-baseline-scale__point" style={{ left: `${valuePosition}%` }} />
      </div>
      <div className="ss-core-metric__foot">
        <span>Baseline {kind === 'accuracy' ? metricPercent(baseline) : kind === 'auc' ? metric(baseline) : '0'}</span>
        {comparison ? <span>{comparison}</span> : null}
      </div>
    </article>
  );
}

function ClassificationMetric({ label, value, accent, info }) {
  return (
    <article className="ss-classification-metric" style={{ '--metric-accent': accent }}>
      <div className="ss-core-metric__head">
        <span>{label} <MetricInfo text={info} /></span>
        <span className="ss-metric-scope">Live</span>
      </div>
      <strong>{metricPercent(value)}</strong>
    </article>
  );
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
  const mccValue = liveN > 0 ? c.mcc : ev?.mcc;
  const aucValue = ev?.roc_auc;
  const totalN = combined?.n ?? c.n ?? 0;
  const evalN = combined?.n_eval ?? ev?.n ?? 0;
  const pendingN = c.pending ?? 0;
  const evalShare = totalN > 0 ? (evalN / totalN) * 100 : 0;
  const liveShare = totalN > 0 ? (liveN / totalN) * 100 : 0;

  return (
    <div>
      <Hero lastRun={health?.last_run} />
      <LastRunBanner lastRun={health?.last_run} />

      <div className="ss-card ss-performance-card">
        <div className="ss-performance-head">
          <div>
            <h2>
              Model performance
              {dashboard.model_type ? <span className="ss-tag">{dashboard.model_type}</span> : null}
            </h2>
            <p>Evaluation and live-monitoring metrics with reference baselines.</p>
          </div>
        </div>

        <div className="ss-metric-group-head">
          <h3>Core metrics</h3>
          <span className="ss-tag">Overall · evaluation + live</span>
        </div>
        <div className="ss-core-metrics">
          <CoreMetric
            label="Accuracy"
            value={accValue}
            displayValue={metricPercent(accValue)}
            kind="accuracy"
            baseline={0.5}
            domain={[0, 1]}
            scope="Overall"
            comparison={ev?.accuracy != null ? `Eval ${metricPercent(ev.accuracy)}` : null}
            info="The share of predictions that matched the actual market direction."
          />
          <CoreMetric
            label="ROC-AUC"
            value={aucValue}
            displayValue={metric(aucValue)}
            kind="auc"
            baseline={0.5}
            domain={[0, 1]}
            scope="Evaluation"
            comparison="Higher is better"
            info="How well the model separates up days from down days across decision thresholds."
          />
          <CoreMetric
            label="MCC"
            value={mccValue}
            displayValue={metric(mccValue)}
            kind="mcc"
            baseline={0}
            domain={[-1, 1]}
            scope={liveN > 0 ? 'Live' : 'Evaluation'}
            comparison={liveN > 0 && ev?.mcc != null ? `Eval ${metric(ev.mcc)}` : 'Range −1 to +1'}
            info="A balanced correlation score from −1 to +1; zero means no predictive relationship."
          />
        </div>

        <div className="ss-metric-group-head ss-metric-group-head--secondary">
          <h3>Classification metrics</h3>
          {liveN > 0 ? <span className="ss-tag">Live monitoring · {liveN} days</span> : null}
        </div>
        <div className="ss-classification-metrics">
          <ClassificationMetric label="Precision" value={liveN > 0 ? c.precision : null} accent="#2dd4bf"
            info="Of the predicted positive days, the share that were actually positive." />
          <ClassificationMetric label="Recall" value={liveN > 0 ? c.recall : null} accent="#a78bfa"
            info="Of the actual positive days, the share the model identified." />
          <ClassificationMetric label="F1" value={liveN > 0 ? c.f1 : null} accent="#fbbf24"
            info="The harmonic mean of precision and recall." />
        </div>

        <div className="ss-metric-status" aria-label="Metric sample status">
          <div className="ss-metric-status__total">
            <span>Sample coverage</span>
            <div><strong>{totalN}</strong><small>observations</small></div>
          </div>

          <div className="ss-metric-status__composition">
            <div className="ss-metric-status__labels">
              <span>
                <i className="is-eval" aria-hidden="true" />
                Evaluation <strong>{evalN}</strong><small>{evalShare.toFixed(1)}%</small>
              </span>
              <span>
                <i className="is-live" aria-hidden="true" />
                Live <strong>{liveN}</strong><small>{liveShare.toFixed(1)}%</small>
              </span>
            </div>
            <div
              className="ss-metric-status__track"
              role="img"
              aria-label={`${evalN} evaluation observations and ${liveN} live observations`}
            >
              <span className="is-eval" style={{ width: `${evalShare}%` }} />
              <span className="is-live" style={{ width: `${liveShare}%` }} />
            </div>
          </div>

          <div className={`ss-metric-status__outcomes ${pendingN === 0 ? 'is-ready' : 'is-pending'}`}>
            <span className="ss-metric-status__icon" aria-hidden="true">
              {pendingN === 0 ? '✓' : '…'}
            </span>
            <div>
              <span>Outcomes</span>
              <strong>{pendingN === 0 ? 'Up to date' : `${pendingN} pending`}</strong>
            </div>
          </div>
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
        <HeadlineList headlines={latest.headlines} initialVisible={24} total={latest.total} />
      </div>
    </div>
  );
}

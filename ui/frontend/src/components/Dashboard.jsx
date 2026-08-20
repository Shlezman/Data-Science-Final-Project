import React, { useEffect, useState, useCallback } from 'react';
import { getJson } from '../lib/api.js';
import { pct, direction, directionCls, outcome, outcomeCls } from '../lib/format.js';
import HeadlineList from './HeadlineList.jsx';
import Hero from './Hero.jsx';
import EdaPanels from './EdaPanels.jsx';

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
  const [perf, setPerf] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      const [d, h, p] = await Promise.all([
        getJson('/api/dashboard'),
        getJson('/api/health'),
        getJson('/api/performance'),
      ]);
      setDashboard(d);
      setHealth(h);
      setPerf(p);
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

  const recent = dashboard.recent || [];
  const latest = dashboard.latest_headlines || {};

  // The Model-performance panel is rendered VERBATIM from /api/performance —
  // the server owns every number/label (models/performance.json overrides).
  const core = perf?.core || [];
  const classification = perf?.classification || [];
  const sample = perf?.sample || { total: 0, eval: 0, live: 0, pending: 0 };
  const totalN = sample.total ?? 0;
  const evalN = sample.eval ?? 0;
  const liveN = sample.live ?? 0;
  const pendingN = sample.pending ?? 0;
  const evalShare = totalN > 0 ? (evalN / totalN) * 100 : 0;
  const liveShare = totalN > 0 ? (liveN / totalN) * 100 : 0;

  return (
    <div>
      <Hero lastRun={health?.last_run} />
      <LastRunBanner lastRun={health?.last_run} />

      <div className="ss-card ss-performance-card">
        <div className="ss-performance-head ss-dashboard-section-head">
          <div className="ss-dashboard-section-head__copy">
            <span className="ss-dashboard-section-head__eyebrow">Model evaluation</span>
            <h2>
              Model performance
              {perf?.model_type ? <span className="ss-tag">{perf.model_type}</span> : null}
            </h2>
            <p>{perf?.subtitle || 'Evaluation and live-monitoring metrics with reference baselines.'}</p>
          </div>
        </div>

        <div className="ss-metric-group-head">
          <h3>Core metrics</h3>
          {perf?.core_tag ? <span className="ss-tag">{perf.core_tag}</span> : null}
        </div>
        <div className="ss-core-metrics">
          {core.map((m) => (
            <CoreMetric
              key={m.label}
              label={m.label}
              value={m.value}
              displayValue={m.kind === 'accuracy' ? metricPercent(m.value) : metric(m.value)}
              kind={m.kind}
              baseline={m.baseline ?? 0.5}
              domain={m.domain || [0, 1]}
              scope={m.scope}
              comparison={m.comparison}
              info={m.info || ''}
            />
          ))}
        </div>

        <div className="ss-metric-group-head ss-metric-group-head--secondary">
          <h3>Classification metrics</h3>
          {perf?.classification_tag ? <span className="ss-tag">{perf.classification_tag}</span> : null}
        </div>
        <div className="ss-classification-metrics">
          {classification.map((m) => (
            <ClassificationMetric key={m.label} label={m.label} value={m.value}
                                  accent={m.accent || '#2dd4bf'} info={m.info || ''} />
          ))}
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

      <div className="ss-card">
        <div className="ss-dashboard-section-head">
          <div className="ss-dashboard-section-head__copy">
            <span className="ss-dashboard-section-head__eyebrow">Prediction history</span>
            <h2>Recent predictions</h2>
            <p>Latest model forecasts compared with observed market direction.</p>
          </div>
        </div>
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
        <div className="ss-dashboard-section-head">
          <div className="ss-dashboard-section-head__copy">
            <span className="ss-dashboard-section-head__eyebrow">News signal</span>
            <h2>Live headlines</h2>
            <p>Scored headlines from the latest available news day.</p>
          </div>
          <div className="ss-dashboard-section-head__meta">
            {latest.date ? <span className="ss-tag">{latest.date}</span> : null}
            {typeof latest.total === 'number' ? (
              <span className="ss-tag">{latest.total} total</span>
            ) : null}
          </div>
        </div>
        <HeadlineList headlines={latest.headlines} initialVisible={24} total={latest.total} />
      </div>
    </div>
  );
}

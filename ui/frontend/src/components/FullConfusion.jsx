import React, { useEffect, useState } from 'react';
import { getJson } from '../lib/api.js';

/**
 * Formats a metric that may be null/NaN into a fixed-precision string.
 *
 * @param {number|null|undefined} v The metric value.
 * @returns {string} Formatted value or "—".
 */
function metric(v) {
  return typeof v === 'number' && !Number.isNaN(v) ? v.toFixed(3) : '—';
}

/**
 * Full-history confusion matrix (scope = all labeled days), read from
 * /api/confusion/full. Renders the same 2×2 grid style as the live matrix plus
 * accuracy/precision/recall/N, tagged "All days" so its scope is explicit.
 *
 * @returns {JSX.Element} The panel.
 */
export default function FullConfusion() {
  const [data, setData] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    getJson('/api/confusion/full')
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setReady(true));
  }, []);

  if (!ready) {
    return <p className="ss-muted">Loading full-history matrix…</p>;
  }
  if (!data || !data.n) {
    return (
      <p className="ss-muted">
        No full-history evaluation yet — run <code>scripts/compute_full_eval.py</code> on the pipeline box.
      </p>
    );
  }

  const c = data;
  return (
    <div>
      <p className="ss-section-title">
        Metrics <span className="ss-tag">All days</span>
        {c.model_version ? <span className="ss-tag">{c.model_version}</span> : null}
      </p>
      <div className="ss-stat-grid">
        <div className="ss-stat"><div className="label">Accuracy</div><div className="value">{metric(c.accuracy)}</div></div>
        <div className="ss-stat"><div className="label">Precision</div><div className="value">{metric(c.precision)}</div></div>
        <div className="ss-stat"><div className="label">Recall</div><div className="value">{metric(c.recall)}</div></div>
        <div className="ss-stat"><div className="label">F1</div><div className="value">{metric(c.f1)}</div></div>
        <div className="ss-stat"><div className="label">MCC</div><div className="value">{metric(c.mcc)}</div></div>
        <div className="ss-stat"><div className="label">N (days)</div><div className="value">{c.n ?? 0}</div></div>
      </div>
    </div>
  );
}

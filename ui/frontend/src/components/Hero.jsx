import React, { useEffect, useState } from 'react';
import { getJson } from '../lib/api.js';

/**
 * Big current-day up/down hero. Reads /api/prediction/today and shows a large
 * green ▲ UP or red ▼ DOWN with the predicted-direction confidence, date, and
 * served-model version. Confidence is the probability of the PREDICTED class
 * (up-prob if up, else 1 − up-prob).
 *
 * @param {object} props Component props.
 * @param {object} [props.lastRun] The `/api/health` `last_run` payload. Its
 *   last-success timestamp joins the metadata here so routine run status sits
 *   with the prediction it belongs to instead of in a banner of its own; the
 *   field is omitted entirely when the orchestrator has not reported one.
 *   Failures stay in the louder `LastRunBanner`.
 * @returns {JSX.Element} The hero card.
 */
export default function Hero({ lastRun }) {
  const [pred, setPred] = useState(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    getJson('/api/prediction/today')
      .then((p) => setPred(p && p.date ? p : null))
      .catch(() => setPred(null))
      .finally(() => setReady(true));
  }, []);

  if (!ready) {
    return <div className="ss-hero ss-hero--pending"><span>Loading current prediction…</span></div>;
  }
  if (!pred) {
    return (
      <div className="ss-hero ss-hero--pending">
        <span>No current-day prediction yet.</span>
      </div>
    );
  }

  const up = pred.up;
  const raw = typeof pred.confidence === 'number' ? pred.confidence : 0.5;
  const dirConf = Math.round((up ? raw : 1 - raw) * 100);

  return (
    <div className={`ss-hero ${up ? 'is-up' : 'is-down'}`}>
      <div className="ss-hero__badge" aria-hidden="true">
        <svg viewBox="0 0 48 48" width="34" height="34">
          <path d={up ? 'M24 15 L36 31 L12 31 Z' : 'M24 33 L12 17 L36 17 Z'}
                fill="currentColor" stroke="currentColor" strokeWidth="5"
                strokeLinejoin="round" />
        </svg>
      </div>

      <div className="ss-hero__body">
        <div className="ss-hero__dir">{up ? 'UP' : 'DOWN'}</div>
        <div className="ss-hero__meter">
          <div className="ss-hero__track" aria-hidden="true">
            <span style={{ width: `${dirConf}%` }} />
          </div>
          <span className="ss-hero__conf">{dirConf}% confidence</span>
        </div>
      </div>

      <dl className="ss-hero__meta">
        <div>
          <dt>Trading day</dt>
          <dd>{pred.date}</dd>
        </div>
        {lastRun?.last_success ? (
          <div>
            <dt>Last run</dt>
            <dd>{lastRun.last_success}</dd>
          </div>
        ) : null}
        {pred.model_version ? (
          <div>
            <dt>Model</dt>
            <dd>{pred.model_version}</dd>
          </div>
        ) : null}
      </dl>
    </div>
  );
}

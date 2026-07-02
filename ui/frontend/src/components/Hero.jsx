import React, { useEffect, useState } from 'react';
import { getJson } from '../lib/api.js';

/**
 * Big current-day up/down hero. Reads /api/prediction/today and shows a large
 * green ▲ UP or red ▼ DOWN with the predicted-direction confidence, date, and
 * served-model version. Confidence is the probability of the PREDICTED class
 * (up-prob if up, else 1 − up-prob).
 *
 * @returns {JSX.Element} The hero card.
 */
export default function Hero() {
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
      <div className="ss-hero__arrow" aria-hidden="true">{up ? '▲' : '▼'}</div>
      <div className="ss-hero__body">
        <div className="ss-hero__dir">{up ? 'UP' : 'DOWN'}</div>
        <div className="ss-hero__conf">{dirConf}% confidence</div>
        <div className="ss-hero__sub">
          {pred.date}{pred.model_version ? ` · ${pred.model_version}` : ''}
        </div>
      </div>
    </div>
  );
}

import React, { useEffect, useState } from 'react';
import { getJson } from '../lib/api.js';
import { pct } from '../lib/format.js';

/**
 * Maps a persona vote to its arrow glyph and color modifier suffix.
 *
 * @param {string} vote One of 'up', 'down' or 'neutral'.
 * @returns {{glyph: string, cls: string}} Arrow character and CSS suffix.
 */
function voteGlyph(vote) {
  if (vote === 'up') {
    return { glyph: '▲', cls: 'pos' };
  }
  if (vote === 'down') {
    return { glyph: '▼', cls: 'neg' };
  }
  return { glyph: '—', cls: 'neutral' };
}

/**
 * One persona chip: source name, vote arrow, mean sentiment and headline count.
 *
 * @param {object} props Component props.
 * @param {object} props.persona Persona row ({source, n, mean_sentiment, vote}).
 * @param {boolean} [props.emphasized] True for the General persona card.
 * @returns {JSX.Element} The chip.
 */
function PersonaChip({ persona, emphasized = false }) {
  const { glyph, cls } = voteGlyph(persona.vote);
  const mean =
    typeof persona.mean_sentiment === 'number' &&
    !Number.isNaN(persona.mean_sentiment)
      ? persona.mean_sentiment.toFixed(2)
      : '—';
  return (
    <div className={`ss-persona${emphasized ? ' ss-persona--general' : ''}`}>
      <span className="ss-persona__head">
        <span className="ss-persona__source">{persona.source}</span>
        <span className={`ss-persona__vote ss-persona__vote--${cls}`}>
          {glyph}
        </span>
      </span>
      <span className="ss-persona__meta">
        <span className={`ss-pill ss-pill--${cls}`}>{mean}</span>
        <span className="ss-muted">{persona.n} headlines</span>
      </span>
    </div>
  );
}

/**
 * Side-by-side row comparing the model's call against the realized outcome.
 *
 * @param {object} props Component props.
 * @param {object|null} props.model Prediction row
 *   ({model_version, prediction, confidence}), or null when absent.
 * @param {boolean|null} props.actual Realized direction, or null if pending.
 * @returns {JSX.Element} The comparison row.
 */
function VerdictRow({ model, actual }) {
  const pred = model ? voteGlyph(model.prediction ? 'up' : 'down') : null;
  // Confidence is stored for the "up" class; flip it for "down" predictions.
  const conf = model
    ? pct(model.prediction ? model.confidence : 1 - model.confidence)
    : null;
  const realized = actual == null ? null : voteGlyph(actual ? 'up' : 'down');

  return (
    <div className="ss-persona-verdicts">
      <div className="ss-persona-verdict">
        <span className="ss-section-title">Model says</span>
        {model ? (
          <span>
            <span className={`ss-persona__vote ss-persona__vote--${pred.cls}`}>
              {pred.glyph}
            </span>{' '}
            {conf} confident
            {model.model_version ? (
              <span className="ss-tag">{model.model_version}</span>
            ) : null}
          </span>
        ) : (
          <span className="ss-muted">no prediction</span>
        )}
      </div>
      <div className="ss-persona-verdict">
        <span className="ss-section-title">Actually happened</span>
        {realized ? (
          <span
            className={`ss-persona__vote ss-persona__vote--${realized.cls}`}
          >
            {realized.glyph}
          </span>
        ) : (
          <span className="ss-muted">not settled yet</span>
        )}
      </div>
    </div>
  );
}

/**
 * "Who says what?" card: per-source persona stances for one day, with the
 * General persona first and emphasized, followed by a model-vs-reality row.
 * Fetches /api/personas whenever the selected date changes.
 *
 * @param {object} props Component props.
 * @param {string} props.date Selected day as 'YYYY-MM-DD' (may be empty).
 * @returns {JSX.Element} The persona panel card.
 */
export default function PersonaPanel({ date }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!date) {
      setData(null);
      setError(null);
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    getJson(`/api/personas?date=${encodeURIComponent(date)}`)
      .then((res) => {
        if (!cancelled) {
          setData(res);
          setError(null);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setData(null);
          setError(err.message);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [date]);

  const personas = data?.personas || [];
  const general = data?.general || null;
  const hasPersonas = personas.length > 0 || general != null;

  return (
    <div className="ss-card">
      <h3>Who says what?</h3>
      <p className="ss-muted">
        Each news provider is treated as a persona; its stance is the mean LLM
        sentiment of its headlines that day (≥ +0.5 bullish, ≤ −0.5 bearish).
      </p>

      {error ? <p className="ss-error-text">Error: {error}</p> : null}
      {loading ? <p className="ss-muted">Loading personas…</p> : null}

      {!loading && !error && !hasPersonas ? (
        <p className="ss-muted">No scored headlines for this date.</p>
      ) : null}

      {hasPersonas ? (
        <>
          <div className="ss-persona-grid">
            {general ? <PersonaChip persona={general} emphasized /> : null}
            {personas.map((p) => (
              <PersonaChip key={p.source} persona={p} />
            ))}
          </div>
          <VerdictRow model={data?.model || null} actual={data?.actual ?? null} />
        </>
      ) : null}
    </div>
  );
}

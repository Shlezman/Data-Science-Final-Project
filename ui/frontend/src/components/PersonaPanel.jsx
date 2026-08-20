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
 * Formats a persona's mean sentiment, or an em dash when the day has no score.
 *
 * @param {object} persona Persona row.
 * @returns {string} Two-decimal score or '—'.
 */
function meanText(persona) {
  return typeof persona.mean_sentiment === 'number' &&
    !Number.isNaN(persona.mean_sentiment)
    ? persona.mean_sentiment.toFixed(2)
    : '—';
}

/**
 * The aggregate persona: one accented row above the per-source list.
 *
 * @param {object} props Component props.
 * @param {object} props.persona The General persona row.
 * @returns {JSX.Element} The aggregate row.
 */
function GeneralPersona({ persona }) {
  const { glyph, cls } = voteGlyph(persona.vote);
  return (
    <div className="ss-persona-general">
      <span className="ss-persona-general__source">{persona.source}</span>
      <span className="ss-persona-general__meta">
        <span className={`ss-pill ss-pill--${cls}`}>{meanText(persona)}</span>
        <span className={`ss-persona__vote ss-persona__vote--${cls}`}>
          {glyph}
        </span>
        <span className="ss-muted">{persona.n} headlines</span>
      </span>
    </div>
  );
}

/**
 * One source persona as a single aligned row: name, score, volume bar, count.
 *
 * This was a bordered two-line card in an auto-fill grid. Eleven of them read as a
 * wall of boxes in which nothing lined up, because the score and the count landed
 * wherever each Hebrew outlet name happened to end. As rows sharing one panel, the
 * columns line up across the whole list and the only chrome left is the panel.
 *
 * The vote arrow went with the card: it restated exactly what the pill's colour
 * already says, since both come from the same voteGlyph() call. The aggregate row
 * above still carries one, so the day's direction stays explicit.
 *
 * @param {object} props Component props.
 * @param {object} props.persona Persona row ({source, n, mean_sentiment, vote}).
 * @param {number} props.maxN Headline count of the loudest source, for the bar scale.
 * @returns {JSX.Element} The row.
 */
function PersonaRow({ persona, maxN }) {
  const { cls } = voteGlyph(persona.vote);
  // Floored so the quietest outlet still reads as a bar and not as a dot.
  const share = maxN > 0 ? Math.max(0.08, persona.n / maxN) : 0;
  return (
    <div
      className="ss-persona-row"
      aria-label={`${persona.source}: mean sentiment ${meanText(persona)}, ${persona.n} headlines`}
    >
      <span className="ss-persona-row__source" title={persona.source}>
        {persona.source}
      </span>
      <span className={`ss-pill ss-pill--${cls}`}>{meanText(persona)}</span>
      {/* Volume, deliberately in neutral grey: colour here already means sentiment,
          and a second colour scale on the same row would read as a second score. */}
      <span className="ss-persona-row__volume" aria-hidden="true">
        <span
          className="ss-persona-row__bar"
          style={{ width: `${(share * 100).toFixed(1)}%` }}
        />
      </span>
      <span className="ss-persona-row__n" aria-hidden="true">{persona.n}</span>
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
  const maxN = personas.reduce((m, p) => Math.max(m, p.n), 0);

  return (
    <div className="ss-card">
      <div className="ss-dashboard-section-head ss-dashboard-section-head--subsection">
        <div className="ss-dashboard-section-head__copy">
          <span className="ss-dashboard-section-head__eyebrow">Source personas</span>
          <h3>Who says what?</h3>
          <p>
            Each provider becomes a persona whose stance reflects the mean LLM
            sentiment of its headlines that day.
          </p>
        </div>
      </div>

      {error ? <p className="ss-error-text">Error: {error}</p> : null}
      {loading ? <p className="ss-muted">Loading personas…</p> : null}

      {!loading && !error && !hasPersonas ? (
        <p className="ss-muted">No scored headlines for this date.</p>
      ) : null}

      {hasPersonas ? (
        <>
          {general ? <GeneralPersona persona={general} /> : null}
          {personas.length > 0 ? (
            <div className="ss-persona-list">
              {personas.map((p) => (
                <PersonaRow key={p.source} persona={p} maxN={maxN} />
              ))}
            </div>
          ) : null}
          <VerdictRow model={data?.model || null} actual={data?.actual ?? null} />
        </>
      ) : null}
    </div>
  );
}

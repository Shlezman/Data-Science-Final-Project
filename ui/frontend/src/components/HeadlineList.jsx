import React from 'react';
import { sentimentBadge } from '../lib/format.js';

const CATEGORIES = [
  ['relevance_politics', 'pol'],
  ['relevance_economy', 'eco'],
  ['relevance_security', 'sec'],
  ['relevance_health', 'hlt'],
  ['relevance_science', 'sci'],
  ['relevance_technology', 'tec'],
];

/**
 * Renders the per-headline LLM relevance scores as compact chips.
 * Only categories the model scored above zero are shown (0 = irrelevant).
 *
 * @param {object} props Component props.
 * @param {object} props.h The headline row (relevance_* columns 0–10 or null).
 * @returns {JSX.Element|null} The chips, or null when unscored.
 */
function ScoreChips({ h }) {
  const chips = CATEGORIES
    .map(([key, label]) => [label, h[key]])
    .filter(([, v]) => typeof v === 'number' && v > 0);
  if (!chips.length) return null;
  return (
    <span className="ss-score-chips">
      {chips.map(([label, v]) => (
        <span key={label} className="ss-score-chip" title={label}>
          {label} {v}
        </span>
      ))}
    </span>
  );
}

/**
 * Renders a list of headlines with source, time, text, a sentiment badge and
 * the per-category relevance scores. Shared by the Dashboard (last-day list)
 * and the Archive tab.
 *
 * @param {object} props Component props.
 * @param {Array<object>} props.headlines Headline rows from the API.
 * @returns {JSX.Element} The rendered list (or an empty-state note).
 */
export default function HeadlineList({ headlines }) {
  if (!headlines || headlines.length === 0) {
    return <p className="ss-muted">No headlines for this day.</p>;
  }

  return (
    <ul className="ss-headline-list">
      {headlines.map((h) => {
        const badge = sentimentBadge(h.global_sentiment);
        return (
          <li key={h.id}>
            <span className="ss-headline-meta">
              {h.source} · {h.hour}
            </span>
            <span className="ss-headline-text">{h.headline}</span>
            <span className="ss-headline-scores">
              <span className={`ss-badge ${badge.cls}`}>{badge.text}</span>
              <ScoreChips h={h} />
              {h.scored === false ? (
                <span className="ss-tag">unscored</span>
              ) : null}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

import React from 'react';
import { sentimentBadge } from '../lib/format.js';

/**
 * Renders a list of headlines with source, time, text and a sentiment badge.
 * Shared by the Dashboard (last-day list) and the Archive tab.
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
            <span>
              <span className={`ss-badge ${badge.cls}`}>{badge.text}</span>
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

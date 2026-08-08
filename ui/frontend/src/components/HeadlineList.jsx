import React, { useEffect, useState } from 'react';
import { sentimentBadge } from '../lib/format.js';

const CATEGORIES = [
  ['relevance_politics', 'pol', 'Politics'],
  ['relevance_economy', 'eco', 'Economy'],
  ['relevance_security', 'sec', 'Security'],
  ['relevance_health', 'hlt', 'Health'],
  ['relevance_science', 'sci', 'Science'],
  ['relevance_technology', 'tec', 'Technology'],
];

function categoryScores(headline) {
  return CATEGORIES
    .map(([key, label, name]) => ({ label, name, value: headline[key] }))
    .filter(({ value }) => typeof value === 'number' && value > 0)
    .sort((a, b) => b.value - a.value);
}

function ScoreChips({ scores }) {
  if (!scores.length) return <span className="ss-muted">None</span>;
  return (
    <span className="ss-score-chips">
      {scores.map(({ label, name, value }) => (
        <span key={label} className="ss-score-chip" title={name}>
          {label} {value}
        </span>
      ))}
    </span>
  );
}

function Fact({ label, children, className = '' }) {
  return (
    <span className={`ss-headline-fact ${className}`}>
      <small>{label}</small>
      <span>{children}</span>
    </span>
  );
}

/**
 * Structured headline cards showing the full title and all category scores,
 * with optional progressive rendering for long dashboard feeds.
 */
export default function HeadlineList({ headlines, initialVisible, total }) {
  const [visibleCount, setVisibleCount] = useState(initialVisible || headlines?.length || 0);

  useEffect(() => {
    setVisibleCount(initialVisible || headlines?.length || 0);
  }, [headlines, initialVisible]);

  if (!headlines || headlines.length === 0) {
    return <p className="ss-muted">No headlines for this day.</p>;
  }

  const visibleHeadlines = headlines.slice(0, visibleCount);
  const canShowMore = visibleCount < headlines.length;

  return (
    <>
      <ul className="ss-headline-list ss-headline-cards">
        {visibleHeadlines.map((h) => {
          const badge = sentimentBadge(h.global_sentiment);
          const scores = categoryScores(h);

          return (
            <li key={h.id} className="ss-headline-card">
              <p className="ss-headline-text" dir="rtl" title={h.headline}>{h.headline}</p>

              <div className="ss-headline-facts">
                <Fact label="Source">{h.source}</Fact>
                <Fact label="Published">{h.hour || '—'}</Fact>
                <Fact label="Sentiment">
                  <span className={`ss-badge ${badge.cls}`}>{badge.text}</span>
                </Fact>
                <Fact label="Category relevance" className="ss-headline-fact--categories">
                  {h.scored === false ? <span className="ss-tag">unscored</span> : (
                    <ScoreChips scores={scores} />
                  )}
                </Fact>
              </div>
            </li>
          );
        })}
      </ul>

      {typeof initialVisible === 'number' ? (
        <div className="ss-headline-load-more">
          <span>
            Showing {Math.min(visibleCount, headlines.length)} of {headlines.length} loaded
            {typeof total === 'number' && total > headlines.length ? ` · ${total} total today` : ''}
          </span>
          {canShowMore ? (
            <button
              type="button"
              className="ss-btn secondary"
              onClick={() => setVisibleCount((count) => Math.min(count + initialVisible, headlines.length))}
            >
              Load {Math.min(initialVisible, headlines.length - visibleCount)} more
            </button>
          ) : null}
        </div>
      ) : null}
    </>
  );
}

/*
 * Pure formatting / classification helpers shared across views.
 * Kept dependency-free and side-effect-free for easy reuse and testing.
 */

/**
 * Formats a 0..1 confidence as an integer percentage string.
 *
 * @param {number|null|undefined} value Confidence in the [0, 1] range.
 * @returns {string} e.g. "73%", or "—" when not a finite number.
 */
export function pct(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return `${Math.round(value * 100)}%`;
}

/**
 * Maps a boolean direction prediction/actual to a readable label.
 *
 * @param {boolean|null|undefined} value True=Up, False=Down, null=unknown.
 * @returns {string} "Up", "Down", or "—".
 */
export function direction(value) {
  if (value === true) {
    return 'Up';
  }
  if (value === false) {
    return 'Down';
  }
  return '—';
}

/**
 * Classifies a global sentiment score into a badge variant.
 *
 * @param {number|null|undefined} sentiment Integer sentiment, -10..+10.
 * @returns {{cls: string, text: string}} CSS class and label text.
 */
export function sentimentBadge(sentiment) {
  if (typeof sentiment !== 'number' || Number.isNaN(sentiment)) {
    return { cls: 'neutral', text: 'n/a' };
  }
  if (sentiment > 0) {
    return { cls: 'pos', text: `+${sentiment}` };
  }
  if (sentiment < 0) {
    return { cls: 'neg', text: `${sentiment}` };
  }
  return { cls: 'neutral', text: '0' };
}

/**
 * Computes the hit/miss label for a recent prediction row.
 *
 * @param {boolean} prediction The predicted direction.
 * @param {boolean|null} actual The realized direction, or null if pending.
 * @returns {string} "Hit", "Miss", or "Pending".
 */
export function outcome(prediction, actual) {
  if (actual === null || actual === undefined) {
    return 'Pending';
  }
  return prediction === actual ? 'Hit' : 'Miss';
}

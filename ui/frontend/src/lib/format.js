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
 * Classifies a direction label into a badge variant for color-coded display.
 *
 * @param {boolean|null|undefined} value True=Up, False=Down, null=unknown.
 * @returns {string} "pos", "neg", or "neutral".
 */
export function directionCls(value) {
  if (value === true) {
    return 'pos';
  }
  if (value === false) {
    return 'neg';
  }
  return 'neutral';
}

/**
 * Classifies a Hit/Miss/Pending outcome into a badge variant.
 *
 * @param {string} label The result of {@link outcome}.
 * @returns {string} "pos" for Hit, "neg" for Miss, "neutral" for Pending.
 */
export function outcomeCls(label) {
  if (label === 'Hit') {
    return 'pos';
  }
  if (label === 'Miss') {
    return 'neg';
  }
  return 'neutral';
}

/**
/**
 * Renders a signed score with an explicit sign and a typographic minus (U+2212),
 * which lines up with the digits instead of sitting high and short like the ASCII
 * hyphen. The single source of this convention: badges and the archive's score
 * filters sit next to each other, so spelling −5 two different ways shows.
 *
 * @param {number} value Integer score.
 * @returns {string} e.g. "+3", "−2", "0".
 */
export function signedScore(value) {
  if (value > 0) {
    return `+${value}`;
  }
  if (value < 0) {
    return `−${Math.abs(value)}`;
  }
  return '0';
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
    return { cls: 'pos', text: signedScore(sentiment) };
  }
  if (sentiment < 0) {
    return { cls: 'neg', text: signedScore(sentiment) };
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

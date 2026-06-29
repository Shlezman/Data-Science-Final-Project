/*
 * Thin fetch wrapper. ALL URLs are relative so the app works identically
 * under the Vite dev proxy and when served same-origin by FastAPI in prod.
 * No hosts are hardcoded anywhere.
 */

/**
 * Performs a GET against a relative API path and parses JSON.
 *
 * @param {string} path Relative path beginning with '/api'.
 * @param {object} [options] Optional config.
 * @param {boolean} [options.allow404] When true, a 404 resolves to null
 *   instead of throwing (used by sim graph/report endpoints).
 * @returns {Promise<object|null>} Parsed JSON body, or null on allowed 404.
 * @throws {Error} If the response status is not ok (and not an allowed 404).
 */
export async function getJson(path, options = {}) {
  const { allow404 = false } = options;
  const res = await fetch(path, { headers: { Accept: 'application/json' } });
  if (res.status === 404 && allow404) {
    return null;
  }
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

/**
 * Extracts a human-readable error message from a failed response.
 *
 * @param {Response} res The failed fetch response.
 * @returns {Promise<string>} A best-effort message, never throws.
 */
async function safeErrorDetail(res) {
  try {
    const body = await res.json();
    return body?.error || body?.detail || '';
  } catch {
    return '';
  }
}

/**
 * Builds the same-origin WebSocket URL for the simulation runner.
 *
 * @returns {string} A ws:// or wss:// URL bound to the current host.
 */
export function simRunSocketUrl() {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${scheme}://${location.host}/ws/sim/run`;
}

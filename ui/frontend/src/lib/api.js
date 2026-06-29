/*
 * Thin fetch wrapper. URLs resolve RELATIVE to where the SPA is served
 * (document.baseURI), so the app works at the site root (FastAPI on :3000)
 * AND under a mounted subpath like a Jupyter proxy (/proxy/3000/). No hosts
 * are hardcoded anywhere.
 */

/**
 * Resolves an API path against the document base so it works at root or under
 * a proxy subpath. ``/api/x`` → ``<baseURI>api/x``.
 *
 * @param {string} path Path beginning with '/api'.
 * @returns {string} An absolute URL rooted at the SPA's served location.
 */
function resolve(path) {
  return new URL(path.replace(/^\//, ''), document.baseURI).toString();
}

/**
 * Performs a GET against an API path and parses JSON.
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
  const res = await fetch(resolve(path), { headers: { Accept: 'application/json' } });
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
 * Builds the WebSocket URL for the simulation runner, relative to where the
 * SPA is served (so it works at root and under a proxy subpath).
 *
 * @returns {string} A ws:// or wss:// URL on the current host + mount path.
 */
export function simRunSocketUrl() {
  const u = new URL('ws/sim/run', document.baseURI);
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
  return u.toString();
}

import React, { useState } from 'react';
import { postJson } from '../lib/api.js';

/**
 * Full-page login gate. Posts the shared password to /api/login; on success
 * the server sets an HttpOnly session cookie and the app re-renders.
 *
 * @param {object} props Component props.
 * @param {Function} props.onOk Called after a successful login.
 * @returns {JSX.Element} The login screen.
 */
export default function Login({ onOk }) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (!password || busy) return;
    setBusy(true);
    setError(null);
    try {
      await postJson('/api/login', { password });
      onOk();
    } catch (err) {
      setError(err.status === 401 ? 'Wrong password.' : err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ss-login">
      <form className="ss-login__card" onSubmit={submit}>
        <h1 className="ss-title">
          <span className="ss-title__senti">Senti</span>Sense
        </h1>
        <p className="ss-muted">Enter the access password to continue.</p>
        <input
          type="password"
          value={password}
          autoFocus
          placeholder="Password"
          autoComplete="current-password"
          onChange={(e) => setPassword(e.target.value)}
        />
        <button className="ss-btn" type="submit" disabled={busy || !password}>
          {busy ? 'Checking…' : 'Enter'}
        </button>
        {error ? <p className="ss-error-text">{error}</p> : null}
      </form>
    </div>
  );
}

import React, { useCallback, useEffect, useState } from 'react';
import { getJson, postJson } from '../lib/api.js';

/**
 * Performance-versions manager (operator view, lives under the Models panel).
 *
 * Versions of the Model-performance JSON are stored in the front machine's
 * MongoDB. The ACTIVE version is what /api/performance serves to the
 * dashboard; no active version = fall back to file/computed values. The
 * editor loads any version's document into a textarea for hand-tuning and
 * saves it back as a NEW version (history is never overwritten).
 *
 * @returns {JSX.Element} The versions manager.
 */
export default function PerfVersions() {
  const [versions, setVersions] = useState(null);
  const [mongoUp, setMongoUp] = useState(true);
  const [editorText, setEditorText] = useState('');
  const [note, setNote] = useState('');
  const [message, setMessage] = useState(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await getJson('/api/performance/versions');
      setVersions(res.versions || []);
      setMongoUp(res.mongo !== false);
    } catch (err) {
      setMessage(err.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const act = useCallback(async (fn, okMsg) => {
    setBusy(true);
    setMessage(null);
    try {
      await fn();
      setMessage(okMsg);
      await load();
    } catch (err) {
      setMessage(err.message);
    } finally {
      setBusy(false);
    }
  }, [load]);

  const snapshotComputed = () => act(
    () => postJson('/api/performance/versions', { note: note || 'snapshot of computed values' }),
    'Saved a new version from the computed values.',
  );

  const openInEditor = async (id) => {
    setBusy(true);
    setMessage(null);
    try {
      const res = await getJson(`/api/performance/versions/${id}`);
      setEditorText(JSON.stringify(res.doc, null, 2));
      setNote(`edited from ${id.slice(-6)}`);
      setMessage(`Loaded version …${id.slice(-6)} into the editor.`);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setBusy(false);
    }
  };

  const saveEdited = () => {
    let doc;
    try {
      doc = JSON.parse(editorText);
    } catch (err) {
      setMessage(`Invalid JSON: ${err.message}`);
      return;
    }
    act(() => postJson('/api/performance/versions', { doc, note }),
        'Saved the edited document as a new version.');
  };

  const activate = (id) => act(
    () => postJson(`/api/performance/versions/${id}/activate`),
    'Version activated — the dashboard now serves it.',
  );

  const deactivate = () => act(
    () => postJson('/api/performance/versions/deactivate'),
    'All versions deactivated — dashboard back to computed values.',
  );

  if (versions === null) {
    return <p className="ss-muted">Loading performance versions…</p>;
  }
  if (!mongoUp) {
    return (
      <p className="ss-muted">
        MongoDB is not configured (set <code>SENTISENSE_MONGO_URL</code> in the UI
        environment) — performance versioning is disabled.
      </p>
    );
  }

  return (
    <div>
      <p className="ss-muted">
        Versioned copies of the dashboard&apos;s Model-performance JSON (stored in MongoDB).
        The <b>active</b> version is what the dashboard serves; deactivate all to fall back
        to live computed values. Load a version into the editor, tune the numbers, and save
        it as a new version.
      </p>

      <div className="ss-controls">
        <button className="ss-btn" onClick={snapshotComputed} disabled={busy}>
          Snapshot computed → new version
        </button>
        <button className="ss-btn secondary" onClick={deactivate} disabled={busy}>
          Deactivate all (use computed)
        </button>
      </div>

      {versions.length === 0 ? (
        <p className="ss-muted">No versions stored yet.</p>
      ) : (
        <table className="ss-table">
          <thead>
            <tr><th>Version</th><th>Note</th><th>Created</th><th>Status</th><th /></tr>
          </thead>
          <tbody>
            {versions.map((v) => (
              <tr key={v.id} style={v.active ? { background: 'var(--ss-pos-bg)' } : undefined}>
                <td><code>…{v.id.slice(-6)}</code></td>
                <td>{v.note || '—'}</td>
                <td>{String(v.created_at).slice(0, 19)}</td>
                <td>{v.active ? <span className="ss-badge pos">Active</span> : '—'}</td>
                <td>
                  <button className="ss-btn secondary" disabled={busy}
                          onClick={() => openInEditor(v.id)}>Edit</button>{' '}
                  {!v.active ? (
                    <button className="ss-btn secondary" disabled={busy}
                            onClick={() => activate(v.id)}>Activate</button>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {editorText ? (
        <div style={{ marginTop: 14 }}>
          <p className="ss-section-title">Editor — saves as a NEW version</p>
          <textarea
            value={editorText}
            onChange={(e) => setEditorText(e.target.value)}
            spellCheck={false}
            style={{ width: '100%', minHeight: 280, fontFamily: 'ui-monospace, Menlo, monospace',
                     fontSize: 12.5, background: 'var(--ss-surface-2)', color: 'var(--ss-fg)',
                     border: '1px solid var(--ss-border)', borderRadius: 8, padding: 10 }}
          />
          <div className="ss-controls" style={{ marginTop: 8 }}>
            <input type="text" value={note} placeholder="version note"
                   onChange={(e) => setNote(e.target.value)}
                   style={{ padding: '8px 10px', borderRadius: 8,
                            border: '1px solid var(--ss-border)',
                            background: 'var(--ss-surface-2)', color: 'var(--ss-fg)' }} />
            <button className="ss-btn" onClick={saveEdited} disabled={busy}>
              Save as new version
            </button>
          </div>
        </div>
      ) : null}

      {message ? <p className="ss-muted">{message}</p> : null}
    </div>
  );
}

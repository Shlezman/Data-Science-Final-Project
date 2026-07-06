import React, { useEffect, useState, useCallback } from 'react';
import { getJson, postJson } from '../lib/api.js';

/**
 * Formats a numeric metric to fixed precision, tolerating null / non-numbers.
 *
 * @param {number|null|undefined} value The metric value.
 * @param {number} [digits] Decimal places to keep.
 * @returns {string} Formatted value or "—".
 */
function num(value, digits = 3) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(digits);
}

/**
 * Formats an integer count, tolerating null / non-numbers.
 *
 * @param {number|null|undefined} value The count.
 * @returns {string} The integer as a string, or "—".
 */
function intOr(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return String(Math.round(value));
}

/**
 * Normalizes a timestamp to a YYYY-MM-DD date string.
 *
 * @param {string|null|undefined} value An ISO-ish timestamp or date string.
 * @returns {string} The date part, or "—" when unparseable.
 */
function dateOnly(value) {
  if (!value) {
    return '—';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value).slice(0, 10);
  }
  return parsed.toISOString().slice(0, 10);
}

/**
 * Renders the ROC-AUC cell: the point estimate plus its confidence interval
 * when both bounds are present.
 *
 * @param {object} props Component props.
 * @param {object} props.model The model row.
 * @returns {JSX.Element} The formatted ROC-AUC.
 */
function RocAuc({ model }) {
  const hasCi =
    typeof model.oos_auc_lo === 'number' && typeof model.oos_auc_hi === 'number';
  return (
    <span>
      {num(model.oos_roc_auc, 4)}
      {hasCi ? (
        <span className="ss-muted">
          {' '}
          [{num(model.oos_auc_lo, 3)}, {num(model.oos_auc_hi, 3)}]
        </span>
      ) : null}
    </span>
  );
}

/**
 * Renders the per-row activation control: an "Active" badge for the current
 * champion, or an activate button (with inline status) for the rest.
 *
 * @param {object} props Component props.
 * @param {object} props.model The model row.
 * @param {boolean} props.busy Whether this row's activation is in flight.
 * @param {string|null} props.rowError This row's activation error, if any.
 * @param {function} props.onActivate Handler invoked with the model version.
 * @returns {JSX.Element} The activation cell contents.
 */
function ActivateCell({ model, busy, rowError, onActivate }) {
  if (model.is_active) {
    const how = model.activated_by === 'manual' ? 'manual' : 'auto';
    return <span className="ss-badge pos">Active ({how})</span>;
  }
  return (
    <span>
      <button
        className="ss-btn secondary"
        disabled={busy}
        onClick={() => onActivate(model.version)}
      >
        {busy ? 'Activating…' : 'Activate'}
      </button>
      {rowError ? (
        <span className="ss-error-text" style={{ marginLeft: 8 }}>
          {rowError}
        </span>
      ) : null}
    </span>
  );
}

/**
 * Renders a single model row.
 *
 * @param {object} props Component props.
 * @param {object} props.model The model row.
 * @param {boolean} props.busy Whether this row's activation is in flight.
 * @param {string|null} props.rowError This row's activation error, if any.
 * @param {function} props.onActivate Handler invoked with the model version.
 * @returns {JSX.Element} The table row.
 */
function ModelRow({ model, busy, rowError, onActivate }) {
  const members = Array.isArray(model.members) ? model.members : null;
  const isEnsemble = model.model_type === 'ensemble';
  const highlight = model.is_active
    ? { background: 'var(--ss-pos-bg)' }
    : undefined;
  return (
    <tr style={highlight}>
      <td>
        <div>
          {model.name || model.version}
          {model.model_type ? (
            <span className="ss-tag">{model.model_type}</span>
          ) : null}
        </div>
        {isEnsemble && members ? (
          <div className="ss-muted" style={{ fontSize: 11 }}>
            Members: {members.join(', ')}
          </div>
        ) : model.artifact_format ? (
          <div className="ss-muted" style={{ fontSize: 11 }}>
            {model.artifact_format}
          </div>
        ) : null}
      </td>
      <td>
        <RocAuc model={model} />
      </td>
      <td>{num(model.oos_mcc, 3)}</td>
      <td>{num(model.oos_accuracy, 3)}</td>
      <td>{intOr(model.oos_n)}</td>
      <td>{dateOnly(model.trained_at)}</td>
      <td>
        <ActivateCell
          model={model}
          busy={busy}
          rowError={rowError}
          onActivate={onActivate}
        />
      </td>
    </tr>
  );
}

const EMPTY_MESSAGE =
  'No models registered yet — run scripts/train_registry.py on the training box.';

/**
 * Orders models by out-of-sample ROC-AUC descending, pushing rows with a
 * missing score to the end.
 *
 * @param {Array<object>} models The unsorted model rows.
 * @returns {Array<object>} A new, sorted array.
 */
function sortByRocAuc(models) {
  return [...models].sort((a, b) => {
    const av = typeof a.oos_roc_auc === 'number' ? a.oos_roc_auc : -Infinity;
    const bv = typeof b.oos_roc_auc === 'number' ? b.oos_roc_auc : -Infinity;
    return bv - av;
  });
}

/**
 * Models view: lists the model registry from /api/models, highlights the
 * active champion, and lets the operator activate any other version via
 * POST /api/models/{version}/activate (re-fetching on success so the active
 * flag moves).
 *
 * @returns {JSX.Element} The model registry viewer.
 */
export default function Models() {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);
  const [busyVersion, setBusyVersion] = useState(null);
  const [rowErrors, setRowErrors] = useState({});

  const load = useCallback(async () => {
    try {
      const res = await getJson('/api/models');
      if (res?.error) {
        setModels([]);
        setError(null);
        return;
      }
      setModels(Array.isArray(res?.models) ? res.models : []);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const activate = useCallback(
    async (version) => {
      setBusyVersion(version);
      setRowErrors((prev) => ({ ...prev, [version]: null }));
      try {
        await postJson(`/api/models/${encodeURIComponent(version)}/activate`);
        await load();
      } catch (err) {
        setRowErrors((prev) => ({ ...prev, [version]: err.message }));
      } finally {
        setBusyVersion(null);
      }
    },
    [load],
  );

  if (error && !models) {
    return <p className="ss-error-text">Failed to load models: {error}</p>;
  }
  if (!models) {
    return <p className="ss-muted">Loading models…</p>;
  }

  const sorted = sortByRocAuc(models);

  return (
    <div className="ss-card">
      <h2>Models</h2>
      {error ? <p className="ss-error-text">Error: {error}</p> : null}
      {sorted.length === 0 ? (
        <p className="ss-muted">{EMPTY_MESSAGE}</p>
      ) : (
        <table className="ss-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>ROC-AUC</th>
              <th>MCC</th>
              <th>Accuracy</th>
              <th>N</th>
              <th>Trained</th>
              <th>Active</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((model) => (
              <ModelRow
                key={model.version}
                model={model}
                busy={busyVersion === model.version}
                rowError={rowErrors[model.version] || null}
                onActivate={activate}
              />
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

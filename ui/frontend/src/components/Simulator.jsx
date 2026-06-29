import React, { useEffect, useState, useCallback, useRef } from 'react';
import { getJson, simRunSocketUrl } from '../lib/api.js';
import CytoscapeGraph from './CytoscapeGraph.jsx';

/**
 * Renders the side panel showing a tapped node's attributes.
 *
 * @param {object} props Component props.
 * @param {object|null} props.node The selected node data, or null.
 * @returns {JSX.Element} The side panel.
 */
function NodePanel({ node }) {
  if (!node) {
    return (
      <div className="ss-side-panel">
        <p className="ss-muted">Tap a node to inspect its attributes.</p>
      </div>
    );
  }
  const attrs = node.attrs || {};
  const keys = Object.keys(attrs);
  return (
    <div className="ss-side-panel">
      <h3>{node.label}</h3>
      <p className="ss-muted">Type: {node.type}</p>
      {keys.length === 0 ? (
        <p className="ss-muted">No attributes.</p>
      ) : (
        <dl>
          {keys.map((k) => (
            <React.Fragment key={k}>
              <dt>{k}</dt>
              <dd>
                {typeof attrs[k] === 'object'
                  ? JSON.stringify(attrs[k])
                  : String(attrs[k])}
              </dd>
            </React.Fragment>
          ))}
        </dl>
      )}
    </div>
  );
}

/**
 * Renders the node-type color legend.
 *
 * @param {object} props Component props.
 * @param {Record<string,string>} props.colors Map of type to hex color.
 * @returns {JSX.Element|null} The legend, or null when empty.
 */
function Legend({ colors }) {
  const entries = Object.entries(colors || {});
  if (entries.length === 0) {
    return null;
  }
  return (
    <div className="ss-legend">
      {entries.map(([type, color]) => (
        <span key={type}>
          <span className="swatch" style={{ background: color }} />
          {type}
        </span>
      ))}
    </div>
  );
}

/**
 * Simulator view: choose mode + cached date to load a graph and report, or
 * run a brand-new simulation over a WebSocket and render the streamed result.
 *
 * @returns {JSX.Element} The simulator.
 */
export default function Simulator() {
  const [modes, setModes] = useState([]);
  const [simDates, setSimDates] = useState([]);
  const [mode, setMode] = useState('');
  const [date, setDate] = useState('');

  const [graph, setGraph] = useState(null);
  const [report, setReport] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [legendColors, setLegendColors] = useState({});
  const [loadError, setLoadError] = useState(null);

  const [runDate, setRunDate] = useState('');
  const [events, setEvents] = useState([]);
  const [running, setRunning] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    getJson('/api/sim/modes')
      .then((res) => {
        const list = res?.modes || [];
        setModes(list);
        if (list.length > 0) {
          setMode(list[0]);
        }
      })
      .catch((err) => setLoadError(err.message));

    getJson('/api/sim/dates')
      .then((res) => {
        const list = res?.dates || [];
        setSimDates(list);
        if (list.length > 0) {
          setDate(list[0]);
        }
      })
      .catch((err) => setLoadError(err.message));
  }, []);

  const loadGraphAndReport = useCallback(async (d, m) => {
    if (!d || !m) {
      return;
    }
    setSelectedNode(null);
    setLegendColors({});
    const params = new URLSearchParams({ date: d, mode: m }).toString();
    try {
      const [g, r] = await Promise.all([
        getJson(`/api/sim/graph?${params}`, { allow404: true }),
        getJson(`/api/sim/report?${params}`, { allow404: true }),
      ]);
      setGraph(g);
      setReport(r);
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
      setGraph(null);
      setReport(null);
    }
  }, []);

  useEffect(() => {
    loadGraphAndReport(date, mode);
  }, [date, mode, loadGraphAndReport]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const runSimulation = useCallback(() => {
    const targetDate = runDate || date;
    if (!targetDate || !mode || running) {
      return;
    }
    setEvents([]);
    setRunning(true);

    const ws = new WebSocket(simRunSocketUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ date: targetDate, mode }));
    };
    ws.onmessage = (msg) => {
      let payload;
      try {
        payload = JSON.parse(msg.data);
      } catch {
        return;
      }
      setEvents((prev) => [...prev, payload]);

      if (payload.event === 'done') {
        if (payload.graph) {
          setSelectedNode(null);
          setLegendColors({});
          setGraph(payload.graph);
        }
        setRunning(false);
        loadGraphAndReport(targetDate, mode);
        ws.close();
      }
      if (payload.event === 'error') {
        setRunning(false);
        ws.close();
      }
    };
    ws.onerror = () => {
      setEvents((prev) => [
        ...prev,
        { event: 'error', message: 'WebSocket connection failed' },
      ]);
      setRunning(false);
    };
    ws.onclose = () => {
      setRunning(false);
    };
  }, [runDate, date, mode, running, loadGraphAndReport]);

  return (
    <div>
      <div className="ss-card">
        <h2>Simulator</h2>
        <div className="ss-controls">
          <label className="ss-field">
            Mode
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              {modes.length === 0 ? <option value="">—</option> : null}
              {modes.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <label className="ss-field">
            Cached date
            <select value={date} onChange={(e) => setDate(e.target.value)}>
              {simDates.length === 0 ? (
                <option value="">No cached sims</option>
              ) : null}
              {simDates.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
        </div>

        {loadError ? <p className="ss-error-text">Error: {loadError}</p> : null}

        <div className="ss-graph-wrap">
          <div style={{ flex: '1 1 480px' }}>
            <CytoscapeGraph
              graph={graph}
              onNodeTap={setSelectedNode}
              onLegend={setLegendColors}
            />
            <Legend colors={legendColors} />
          </div>
          <NodePanel node={selectedNode} />
        </div>
      </div>

      <div className="ss-card">
        <h3>Report</h3>
        {report?.report_md ? (
          <pre className="ss-report">{report.report_md}</pre>
        ) : (
          <p className="ss-muted">No report for this date / mode.</p>
        )}
      </div>

      <div className="ss-card">
        <h3>Run new simulation</h3>
        <div className="ss-controls">
          <label className="ss-field">
            Date
            <input
              type="date"
              value={runDate}
              onChange={(e) => setRunDate(e.target.value)}
            />
          </label>
          <button
            className="ss-btn"
            onClick={runSimulation}
            disabled={running || (!runDate && !date) || !mode}
          >
            {running ? 'Running…' : 'Run new simulation'}
          </button>
        </div>
        {events.length > 0 ? (
          <ul className="ss-events">
            {events.map((ev, i) => (
              <li key={i} className={`ev-${ev.event}`}>
                {ev.event === 'running'
                  ? `running… ${ev.elapsed_s ?? '?'}s`
                  : ev.event === 'done'
                    ? `done${ev.cached ? ' (cached)' : ''}`
                    : ev.event === 'error'
                      ? `error: ${ev.message ?? 'unknown'}`
                      : ev.event}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </div>
  );
}

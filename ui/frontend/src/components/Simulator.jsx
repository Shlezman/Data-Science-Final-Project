import React, { useEffect, useState, useCallback, useRef } from 'react';
import { getJson, postJson, simRunSocketUrl } from '../lib/api.js';
import CytoscapeGraph from './CytoscapeGraph.jsx';
import PersonaPanel from './PersonaPanel.jsx';
import AnalystPanel from './AnalystPanel.jsx';

// The API returns the runner's internal mode identifiers. Spelling them out here
// keeps the dropdown readable without the backend carrying display strings.
const MODE_LABELS = {
  source: 'By source',
  flat: 'Pooled',
};
const MODE_HINTS = {
  source: 'One agent per news outlet, so each source argues its own line.',
  flat: 'The whole day’s news pooled and deduped, with no source attribution.',
};

/**
 * Base direction for a report line, by whichever script dominates it.
 *
 * dir="auto" is not usable here: it keys off the FIRST strong character, so a
 * persona line that is almost entirely English flipped to RTL just because the
 * outlet's name is Hebrew — and neighbouring lines disagreed with each other
 * depending on whether the outlet was spelled "Ynet" or "מעריב".
 *
 * @param {string} text One line of the report.
 * @returns {'rtl'|'ltr'} The base direction to render it with.
 */
function lineDir(text) {
  const hebrew = (text.match(/[֐-׿]/g) || []).length;
  const latin = (text.match(/[A-Za-z]/g) || []).length;
  return hebrew > latin ? 'rtl' : 'ltr';
}

/** Splits '**bold**' runs into <strong>, leaving everything else as plain text. */
function inlineBold(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) => (
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={i}>{part.slice(2, -2)}</strong>
      : part
  ));
}

/**
 * Renders the report's Markdown subset: '##' headings, paragraphs and '**bold**'.
 *
 * The generator emits only those three, so this stays short instead of pulling in a
 * Markdown dependency. It replaces a <pre> that printed the source verbatim, which
 * delivered the narrative payoff of the whole tab as literal '##' and '**' in a
 * monospace column.
 *
 * @param {object} props Component props.
 * @param {string} props.md Raw report Markdown.
 * @returns {JSX.Element} Rendered prose.
 */
function ReportBody({ md }) {
  // Line by line, not paragraph by paragraph: the generator does not reliably put a
  // blank line between a heading and its body, and splitting on blank lines swallowed
  // whole sections into the heading.
  return (
    <div className="ss-report">
      {md.split('\n').map((raw, i) => {
        const line = raw.trim();
        if (!line) {
          return null;
        }
        const heading = line.match(/^(#{1,6})\s+(.*)$/);
        if (heading) {
          return <h4 key={i} className="ss-report__heading">{heading[2]}</h4>;
        }
        // Per line, because a Hebrew persona name can head an English analysis.
        return (
          <p key={i} className="ss-report__para" dir={lineDir(line)}>{inlineBold(line)}</p>
        );
      })}
    </div>
  );
}

/**
 * Renders the side panel showing a tapped node's attributes.
 *
 * @param {object} props Component props.
 * @param {object|null} props.node The selected node data, or null.
 * @returns {JSX.Element|null} The side panel, or nothing when no node is selected.
 */
function NodePanel({ node }) {
  // Renders nothing when idle. It used to hold a 340px column saying "tap a node",
  // which sat empty beside the graph for the whole visit and repeated the
  // instruction already in the section description above it.
  if (!node) {
    return null;
  }
  const attrs = node.attrs || {};
  const keys = Object.keys(attrs).filter((k) => k !== 'statement');
  return (
    <div className="ss-side-panel">
      <h3>{node.label}</h3>
      <p className="ss-muted">Type: {node.type}</p>
      {attrs.statement ? (
        <blockquote className="ss-persona-quote">“{attrs.statement}”</blockquote>
      ) : null}
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
  const [simLive, setSimLive] = useState(null);
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

    // Gate live runs on MiroFish reachability; cached graphs render regardless.
    getJson('/api/sim/health')
      .then((res) => setSimLive(res))
      .catch(() => setSimLive({ reachable: false, reason: 'health check failed' }));
  }, []);

  const liveDisabled = simLive != null && !simLive.reachable;

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
        <div className="ss-dashboard-section-head">
          <div className="ss-dashboard-section-head__copy">
            <span className="ss-dashboard-section-head__eyebrow">Narrative simulation</span>
            <h2>Simulator</h2>
            <p>
              Agent personas discuss the day&apos;s news, revealing influence paths
              and the resulting narrative summary.
            </p>
          </div>
        </div>
        {/* Same panel treatment as the archive toolbar. These were bare fields
            floating on the card while the archive's sat in a bordered shell. */}
        <div className="ss-toolbar">
          <div className="ss-toolbar__row">
            <label className="ss-field ss-field--sort">
              Simulation mode
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                {modes.length === 0 ? <option value="">—</option> : null}
                {modes.map((m) => (
                  <option key={m} value={m}>
                    {MODE_LABELS[m] || m}
                  </option>
                ))}
              </select>
            </label>
            <label className="ss-field ss-field--sort">
              Day
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
            {MODE_HINTS[mode] ? (
              <p className="ss-toolbar__hint">
                {MODE_HINTS[mode]}
                {liveDisabled ? ' Simulations are generated automatically after each trading day.' : ''}
              </p>
            ) : null}
          </div>
        </div>

        {loadError ? <p className="ss-error-text">Error: {loadError}</p> : null}
      </div>

      <AnalystPanel date={date} />

      <PersonaPanel date={date} />

      <div className="ss-card">
        <div className="ss-dashboard-section-head ss-dashboard-section-head--subsection">
          <div className="ss-dashboard-section-head__copy">
            <span className="ss-dashboard-section-head__eyebrow">Influence network</span>
            <h3>Agent map</h3>
            <p>
              Node size reflects publishing volume, color shows stance and links show
              agreement or disagreement. Tap a node to inspect its statement.
            </p>
          </div>
          <div className="ss-dashboard-section-head__meta">
            {graph?.meta?.lean ? (
              <span className={`ss-badge ${graph.meta.lean === 'UP' ? 'pos' : graph.meta.lean === 'DOWN' ? 'neg' : 'neutral'}`}>
                Lean {graph.meta.lean}
              </span>
            ) : null}
            {graph?.meta?.n_agents ? (
              <span className="ss-tag">{graph.meta.n_agents} agents</span>
            ) : null}
          </div>
        </div>
        {graph?.meta?.consensus ? (
          <blockquote className="ss-persona-quote">{graph.meta.consensus}</blockquote>
        ) : null}
        <div className="ss-graph-wrap">
          <div className="ss-graph-col">
            <CytoscapeGraph
              graph={graph}
              onNodeTap={setSelectedNode}
              onLegend={setLegendColors}
              emptyMessage="No cached simulation for this day — pick another date or run a new one below."
            />
            <Legend colors={legendColors} />
          </div>
          <NodePanel node={selectedNode} />
        </div>
      </div>

      <div className="ss-card">
        <div className="ss-dashboard-section-head ss-dashboard-section-head--subsection">
          <div className="ss-dashboard-section-head__copy">
            <span className="ss-dashboard-section-head__eyebrow">Narrative output</span>
            <h3>Report</h3>
            <p>Generated synthesis for the selected day and simulation mode.</p>
          </div>
        </div>
        {report?.report_md ? (
          <ReportBody md={report.report_md} />
        ) : (
          <p className="ss-muted">No report for this date / mode.</p>
        )}
      </div>

      {/* Dropped entirely when live runs are off — MiroFish is deliberately absent in
          production, so this card was permanently a heading, one sentence and no
          controls. Its remaining information now rides along with the mode hint. */}
      {liveDisabled ? null : (
      <div className="ss-card">
        <div className="ss-dashboard-section-head ss-dashboard-section-head--subsection">
          <div className="ss-dashboard-section-head__copy">
            <span className="ss-dashboard-section-head__eyebrow">Simulation control</span>
            <h3>Run new simulation</h3>
            <p>Runs the multi-agent simulation for the selected day on the MiroFish service.</p>
          </div>
        </div>
        {(
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
        )}
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
      )}
    </div>
  );
}

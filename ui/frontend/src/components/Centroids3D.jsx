import React, { useEffect, useMemo, useRef, useState } from 'react';
import { getJson } from '../lib/api.js';
import { Plot, darkLayout, chartTheme, PLOT_CONFIG, UP, ACCENT } from '../lib/plotly.js';

const CLUSTER_COLORS = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24',
                        '#a78bfa', '#f87171', '#2dd4bf', '#fb923c'];

/**
 * Detects WebGL availability. Plotly scatter3d REQUIRES WebGL; corporate
 * browsers / remote-desktop sessions often have it disabled, so the drawer
 * degrades to a rotatable software-3D projection instead of an error screen.
 *
 * @returns {boolean} True when a WebGL context can be created.
 */
function detectWebGL() {
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
  } catch {
    return false;
  }
}

/**
 * Software-3D fallback: orthographically projects scatter3d traces onto the
 * screen for a given azimuth/elevation, as plain SVG scatter — an interactive
 * 3D view (rotate via sliders) that needs NO WebGL.
 *
 * @param {Array} traces Plotly scatter3d traces.
 * @param {number} azDeg Azimuth rotation (degrees, about the vertical axis).
 * @param {number} elDeg Elevation/tilt (degrees; 0 = side view, 90 = top view).
 * @returns {Array} Equivalent 2D scatter traces of the rotated projection.
 */
function project3(traces, azDeg, elDeg) {
  const az = (azDeg * Math.PI) / 180;
  const el = (elDeg * Math.PI) / 180;
  const ca = Math.cos(az); const sa = Math.sin(az);
  const ce = Math.cos(el); const se = Math.sin(el);
  return traces.map((t) => {
    const xs = []; const ys = [];
    for (let i = 0; i < t.x.length; i += 1) {
      const x1 = ca * t.x[i] + sa * t.y[i];          // rotate about the z (vertical) axis
      const d = -sa * t.x[i] + ca * t.y[i];          // depth after azimuth spin
      xs.push(x1);
      ys.push(ce * t.z[i] + se * d);                 // tilt: blend height with depth
    }
    return {
      ...t,
      type: 'scatter',
      x: xs, y: ys, z: undefined,
      marker: { ...t.marker, size: Math.max((t.marker?.size ?? 4) * 1.8, 6) },
    };
  });
}

/**
 * The KMeans cluster-center trace (labeled open diamonds), on the chosen axes.
 *
 * @param {Array} centers Projected centers [{id, v: [n_pca]}].
 * @param {number[]} axes The [x,y,z] component indices.
 * @param {string} textColor Label colour for the active theme. Passed in rather than
 *   read here, because this runs inside a useMemo that would otherwise cache the
 *   colour from whichever theme was active when the traces were first built.
 * @returns {object|null} A Plotly trace, or null when no centers stored.
 */
function centersTrace(centers, axes = [0, 1, 2], textColor = '#c9d1d9') {
  if (!centers.length) return null;
  const [ax, ay, az] = axes;
  return {
    type: 'scatter3d', mode: 'markers+text', name: 'Cluster centers',
    x: centers.map((c) => c.v[ax]), y: centers.map((c) => c.v[ay]), z: centers.map((c) => c.v[az]),
    text: centers.map((c) => `K${c.id}`),
    textposition: 'top center',
    // Was a fixed near-white, which vanished against the light card.
    textfont: { size: 11, color: textColor },
    hovertemplate: 'KMeans center %{text}<extra></extra>',
    marker: {
      size: 12, symbol: 'diamond-open', opacity: 1,
      color: centers.map((c) => CLUSTER_COLORS[c.id % CLUSTER_COLORS.length]),
      line: { width: 3 },
    },
  };
}

/**
 * All-days trace: every trading day's news centroid as a green point (no
 * outcome coloring), hover shows date, cluster membership, and volume.
 *
 * @param {Array} pts Day points [{x,y,z,date,n_headlines,cluster}].
 * @returns {object|null} A Plotly trace, or null if empty.
 */
function daysTrace(pts) {
  if (!pts.length) return null;
  return {
    type: 'scatter3d', mode: 'markers', name: 'Day centroids',
    x: pts.map((p) => p.x), y: pts.map((p) => p.y), z: pts.map((p) => p.z),
    text: pts.map((p) => `${p.date}${p.cluster != null ? ` · cluster ${p.cluster}` : ''}`
      + ` · ${p.n_headlines} headlines`),
    customdata: pts.map((p) => p.date),
    hovertemplate: '%{text}<extra></extra>',
    marker: { size: 4, color: UP, opacity: 0.8 },
  };
}

/**
 * Single-day traces: the day's headline vectors (green), the day centroid
 * (accent diamond), and the KMeans cluster centers — on the chosen axes.
 *
 * @param {object} day The /api/centroids/day payload.
 * @param {number[]} axes The [x,y,z] component indices.
 * @param {Array} centers Projected cluster centers.
 * @param {string} textColor Label colour for the active theme.
 * @returns {Array} Plotly traces.
 */
function headlineTraces(day, axes, centers = [], textColor) {
  const [ax, ay, az] = axes;
  const pts = day.points || [];
  const traces = [];
  if (pts.length) {
    traces.push({
      type: 'scatter3d', mode: 'markers', name: 'Headlines',
      x: pts.map((p) => p.v[ax]), y: pts.map((p) => p.v[ay]), z: pts.map((p) => p.v[az]),
      text: pts.map((p) => `[${p.source}] ${String(p.headline).slice(0, 90)}`),
      hovertemplate: '%{text}<extra></extra>',
      marker: { size: 3.5, color: UP, opacity: 0.75 },
    });
  }
  if (day.centroid) {
    traces.push({
      type: 'scatter3d', mode: 'markers', name: 'Day centroid',
      x: [day.centroid[ax]], y: [day.centroid[ay]], z: [day.centroid[az]],
      text: [`Centroid of ${day.date} — the embpca features the model sees`],
      hovertemplate: '%{text}<extra></extra>',
      marker: { size: 10, color: ACCENT, symbol: 'diamond', opacity: 1 },
    });
  }
  const kc = centersTrace(centers, axes, textColor);
  if (kc) traces.push(kc);
  return traces;
}

/**
 * Right-side sliding drawer with two 3D views:
 *  - "All days": one green point per trading day (its news centroid), the
 *    KMeans cluster centers as labeled diamonds, a time slider, and
 *    click-through to any day's headline cloud.
 *  - "Single day": every headline of that day projected through the SAME
 *    leak-safe scaler→PCA basis the dataset features use (16 dims — pick any
 *    three as axes), with the day centroid and the cluster centers marked.
 * Falls back to a rotatable software-3D projection when WebGL is disabled.
 *
 * Controlled by the caller: the trigger lives in the app header, so the drawer
 * is reachable from every tab rather than only from the Dashboard.
 *
 * @param {boolean} open Whether the drawer is showing.
 * @param {Function} onClose Called when the drawer asks to close.
 * @returns {JSX.Element} The drawer.
 */
export default function Centroids3D({ open, onClose }) {
  const [points, setPoints] = useState([]);
  const [upto, setUpto] = useState(0);
  const [view, setView] = useState('all');
  const [dayDate, setDayDate] = useState('');
  const [day, setDay] = useState(null);
  const [dayLoading, setDayLoading] = useState(false);
  const [axes, setAxes] = useState([0, 1, 2]);
  const [clusters, setClusters] = useState([]);
  const [rot, setRot] = useState([35, 55]);        // [azimuth°, elevation°] for software 3D
  const webgl = useMemo(detectWebGL, []);
  const fetched = useRef(false);

  // Loaded on FIRST open, not on mount. The drawer now mounts with the app shell
  // rather than with the Dashboard, so an eager fetch would put a centroid request
  // on every page load for a panel most visits never open.
  useEffect(() => {
    if (!open || fetched.current) return;
    fetched.current = true;
    getJson('/api/centroids')
      .then((d) => {
        const pts = d?.points || [];
        setPoints(pts);
        setClusters(d?.clusters || []);
        setUpto(pts.length ? pts.length - 1 : 0);
        if (pts.length && !dayDate) setDayDate(pts[pts.length - 1].date);
      })
      .catch(() => setPoints([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (view !== 'day' || !dayDate) return;
    setDayLoading(true);
    getJson(`/api/centroids/day?date=${dayDate}`)
      .then(setDay)
      .catch((e) => setDay({ points: [], error: e.message }))
      .finally(() => setDayLoading(false));
  }, [view, dayDate]);

  // Gridlines were fixed white — invisible on the light card. The 3D scene axes are
  // outside darkLayout's xaxis/yaxis merge, so they need the palette explicitly.
  const ct = chartTheme();

  const shown = useMemo(() => points.slice(0, upto + 1), [points, upto]);
  // ct.text is a dependency: without it the memo served traces carrying the label
  // colour of whichever theme was active when they were first built.
  const allTraces = useMemo(
    () => [daysTrace(shown), centersTrace(clusters, [0, 1, 2], ct.text)].filter(Boolean),
    [shown, clusters, ct.text],
  );

  const nPca = day?.n_pca || 16;
  const sceneAxis = (title) => ({ title, gridcolor: ct.grid, color: ct.text });
  const axisTitles = view === 'all'
    ? ['pca-0', 'pca-1', 'pca-2']
    : [`pca-${axes[0]}`, `pca-${axes[1]}`, `pca-${axes[2]}`];
  const layout = darkLayout(webgl
    ? {
      showlegend: true,
      legend: { orientation: 'h', y: -0.05 },
      scene: { xaxis: sceneAxis(axisTitles[0]), yaxis: sceneAxis(axisTitles[1]),
               zaxis: sceneAxis(axisTitles[2]) },
      margin: { l: 0, r: 0, t: 0, b: 0 },
    }
    : {
      showlegend: true,
      legend: { orientation: 'h', y: -0.12 },
      xaxis: { title: '', gridcolor: ct.grid, zeroline: false },
      yaxis: { title: '', gridcolor: ct.grid, zeroline: false, scaleanchor: 'x' },
      margin: { l: 36, r: 12, t: 10, b: 30 },
    });
  const asPlot = (traces) => (webgl ? traces : project3(traces, rot[0], rot[1]));

  const rotationControls = webgl ? null : (
    <div className="ss-controls" style={{ marginBottom: 6, alignItems: 'center' }}>
      <label className="ss-field" style={{ minWidth: 180 }}>
        Rotate ({rot[0]}°)
        <input type="range" min={0} max={360} value={rot[0]}
               onChange={(e) => setRot([Number(e.target.value), rot[1]])} />
      </label>
      <label className="ss-field" style={{ minWidth: 180 }}>
        Tilt ({rot[1]}°)
        <input type="range" min={0} max={90} value={rot[1]}
               onChange={(e) => setRot([rot[0], Number(e.target.value)])} />
      </label>
      <span className="ss-muted">software 3D (WebGL off) — drag sliders to rotate</span>
    </div>
  );

  const openDay = (date) => { setDayDate(date); setView('day'); };

  return (
    <>
      <div className={`ss-drawer ${open ? 'is-open' : ''}`} aria-hidden={!open}>
        <div className="ss-drawer__head">
          <span>Daily news centroids (3D)</span>
          <span style={{ display: 'flex', gap: 6 }}>
            <button className={`ss-btn ${view === 'all' ? '' : 'ss-btn--ghost'}`}
                    onClick={() => setView('all')}>All days</button>
            <button className={`ss-btn ${view === 'day' ? '' : 'ss-btn--ghost'}`}
                    onClick={() => setView('day')}>Single day</button>
            <button className="ss-drawer__close" onClick={onClose} aria-label="Close">×</button>
          </span>
        </div>

        {view === 'all' ? (
          points.length === 0 ? (
            <p className="ss-muted" style={{ padding: '0 16px' }}>
              No centroid data — needs <code>daily_embedding_derived</code>.
            </p>
          ) : (
            <div className="ss-drawer__body">
              <p className="ss-muted" style={{ margin: '4px 2px 8px' }}>
                Each green point is one trading day&apos;s news centroid; the open diamonds are
                the KMeans cluster centers (the clustering behind the <code>embclus_dist</code>
                features). Click a day to open its headline cloud.
              </p>
              {rotationControls}
              <Plot data={asPlot(allTraces)} layout={layout} config={PLOT_CONFIG}
                    style={{ width: '100%', height: '64vh' }} useResizeHandler
                    onClick={(ev) => {
                      const d = ev?.points?.[0]?.customdata;
                      if (d) openDay(d);
                    }} />
              <div className="ss-drawer__slider">
                <input type="range" min={0} max={Math.max(points.length - 1, 0)} value={upto}
                       onChange={(e) => setUpto(Number(e.target.value))} />
                <span className="ss-muted">
                  through {points[upto]?.date} · {shown.length}/{points.length} days
                </span>
              </div>
            </div>
          )
        ) : (
          <div className="ss-drawer__body">
            <p className="ss-muted" style={{ margin: '4px 2px 8px' }}>
              Every headline of the chosen day (green), projected into the same 16-dim PCA space
              as the model&apos;s <code>embpca</code> features. The blue diamond is the day
              centroid; the open diamonds are the KMeans cluster centers.
              {(() => {
                const c = points.find((p) => p.date === dayDate)?.cluster;
                return c != null ? (
                  <span> This day belongs to <b style={{ color: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }}>
                    cluster {c}</b>.</span>
                ) : null;
              })()}
            </p>
            <div className="ss-controls" style={{ marginBottom: 8 }}>
              <label className="ss-field">
                Date
                <input type="date" value={dayDate} onChange={(e) => setDayDate(e.target.value)} />
              </label>
              {['X', 'Y', 'Z'].map((lbl, i) => (
                <label className="ss-field" key={lbl}>
                  {lbl} axis
                  <select value={axes[i]}
                          onChange={(e) => {
                            const next = axes.slice();
                            next[i] = Number(e.target.value);
                            setAxes(next);
                          }}>
                    {Array.from({ length: nPca }, (_, k) => (
                      <option key={k} value={k}>pca-{k}</option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
            {dayLoading ? (
              <p className="ss-muted">Loading day cloud…</p>
            ) : !day || !(day.points || []).length ? (
              <p className="ss-muted">
                {day?.error || 'No data for that date.'}
              </p>
            ) : (
              <>
                {rotationControls}
                <Plot data={asPlot(headlineTraces(day, axes, clusters, ct.text))} layout={layout} config={PLOT_CONFIG}
                      style={{ width: '100%', height: '62vh' }} useResizeHandler />
                <p className="ss-muted" style={{ margin: '6px 2px 0' }}>
                  {day.points.length} headlines · hover a point for its text and source.
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );
}

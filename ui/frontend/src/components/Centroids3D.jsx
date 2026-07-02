import React, { useEffect, useMemo, useState } from 'react';
import { getJson } from '../lib/api.js';
import { Plot, darkLayout, PLOT_CONFIG, UP, DOWN, NEUTRAL, ACCENT } from '../lib/plotly.js';

/**
 * Builds a colored 3D scatter trace for a subset of day-centroids.
 *
 * @param {Array} pts Points [{x,y,z,date,n_headlines,actual}].
 * @param {boolean|null} actual Which class this trace holds (null = unknown).
 * @param {string} name Legend label.
 * @param {string} color Marker color.
 * @returns {object|null} A Plotly trace, or null if empty.
 */
function dayTrace(pts, actual, name, color) {
  const sel = pts.filter((p) => p.actual === actual);
  if (!sel.length) return null;
  return {
    type: 'scatter3d', mode: 'markers', name,
    x: sel.map((p) => p.x), y: sel.map((p) => p.y), z: sel.map((p) => p.z),
    text: sel.map((p) => `${p.date} · ${p.n_headlines} headlines`),
    customdata: sel.map((p) => p.date),
    hovertemplate: '%{text}<extra></extra>',
    marker: { size: 4, color, opacity: 0.85 },
  };
}

/**
 * Splits one day's projected headlines into sentiment-colored traces plus the
 * centroid marker, using the chosen 3 of the 16 embpca axes.
 *
 * @param {object} day The /api/centroids/day payload.
 * @param {number[]} axes The [x,y,z] component indices.
 * @returns {Array} Plotly traces.
 */
function headlineTraces(day, axes) {
  const [ax, ay, az] = axes;
  const groups = [
    { name: 'Positive', color: UP, test: (s) => typeof s === 'number' && s > 0 },
    { name: 'Negative', color: DOWN, test: (s) => typeof s === 'number' && s < 0 },
    { name: 'Neutral / unscored', color: NEUTRAL, test: (s) => !(typeof s === 'number' && s !== 0) },
  ];
  const traces = groups.map((g) => {
    const sel = (day.points || []).filter((p) => g.test(p.sentiment));
    if (!sel.length) return null;
    return {
      type: 'scatter3d', mode: 'markers', name: g.name,
      x: sel.map((p) => p.v[ax]), y: sel.map((p) => p.v[ay]), z: sel.map((p) => p.v[az]),
      text: sel.map((p) => `[${p.source}] ${String(p.headline).slice(0, 90)}`
        + (typeof p.sentiment === 'number' ? ` (sent ${p.sentiment})` : '')),
      hovertemplate: '%{text}<extra></extra>',
      marker: { size: 3.5, color: g.color, opacity: 0.75 },
    };
  }).filter(Boolean);
  if (day.centroid) {
    traces.push({
      type: 'scatter3d', mode: 'markers', name: 'Day centroid',
      x: [day.centroid[ax]], y: [day.centroid[ay]], z: [day.centroid[az]],
      text: [`Centroid of ${day.date} — the embpca features the model sees`],
      hovertemplate: '%{text}<extra></extra>',
      marker: { size: 10, color: ACCENT, symbol: 'diamond', opacity: 1 },
    });
  }
  return traces;
}

/**
 * Right-side sliding drawer with two 3D views:
 *  - "All days": one point per trading day (its news centroid, embpca 0/1/2),
 *    green/red by the realised up/down; a slider sweeps time; clicking a day
 *    jumps to its single-day view.
 *  - "Single day": every headline of that day projected through the SAME
 *    leak-safe scaler→PCA basis the dataset features use (16 dims — pick any
 *    three as axes), colored by LLM sentiment, with the day centroid marked.
 *
 * @returns {JSX.Element} The toggle button + drawer.
 */
export default function Centroids3D() {
  const [points, setPoints] = useState([]);
  const [open, setOpen] = useState(false);
  const [upto, setUpto] = useState(0);
  const [view, setView] = useState('all');
  const [dayDate, setDayDate] = useState('');
  const [day, setDay] = useState(null);
  const [dayLoading, setDayLoading] = useState(false);
  const [axes, setAxes] = useState([0, 1, 2]);

  useEffect(() => {
    getJson('/api/centroids')
      .then((d) => {
        const pts = d?.points || [];
        setPoints(pts);
        setUpto(pts.length ? pts.length - 1 : 0);
        if (pts.length && !dayDate) setDayDate(pts[pts.length - 1].date);
      })
      .catch(() => setPoints([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (view !== 'day' || !dayDate) return;
    setDayLoading(true);
    getJson(`/api/centroids/day?date=${dayDate}`)
      .then(setDay)
      .catch((e) => setDay({ points: [], error: e.message }))
      .finally(() => setDayLoading(false));
  }, [view, dayDate]);

  const shown = useMemo(() => points.slice(0, upto + 1), [points, upto]);
  const allTraces = useMemo(
    () => [
      dayTrace(shown, true, 'Up', UP),
      dayTrace(shown, false, 'Down', DOWN),
      dayTrace(shown, null, 'Unsettled', NEUTRAL),
    ].filter(Boolean),
    [shown],
  );

  const nPca = day?.n_pca || 16;
  const sceneAxis = (title) => ({ title, gridcolor: 'rgba(255,255,255,0.1)' });
  const layout = darkLayout({
    showlegend: true,
    legend: { orientation: 'h', y: -0.05 },
    scene: view === 'all'
      ? { xaxis: sceneAxis('pca-0'), yaxis: sceneAxis('pca-1'), zaxis: sceneAxis('pca-2') }
      : { xaxis: sceneAxis(`pca-${axes[0]}`), yaxis: sceneAxis(`pca-${axes[1]}`), zaxis: sceneAxis(`pca-${axes[2]}`) },
    margin: { l: 0, r: 0, t: 0, b: 0 },
  });

  const openDay = (date) => { setDayDate(date); setView('day'); };

  return (
    <>
      <button className="ss-drawer__toggle" onClick={() => setOpen(true)}>
        3D centroids
      </button>
      <div className={`ss-drawer ${open ? 'is-open' : ''}`}>
        <div className="ss-drawer__head">
          <span>Daily news centroids (3D)</span>
          <span style={{ display: 'flex', gap: 6 }}>
            <button className={`ss-btn ${view === 'all' ? '' : 'ss-btn--ghost'}`}
                    onClick={() => setView('all')}>All days</button>
            <button className={`ss-btn ${view === 'day' ? '' : 'ss-btn--ghost'}`}
                    onClick={() => setView('day')}>Single day</button>
            <button className="ss-drawer__close" onClick={() => setOpen(false)}>×</button>
          </span>
        </div>

        {view === 'all' ? (
          points.length === 0 ? (
            <p className="ss-muted" style={{ padding: '0 16px' }}>
              No centroid data — needs <code>daily_embedding_derived</code> + <code>champion_full_eval</code>.
            </p>
          ) : (
            <div className="ss-drawer__body">
              <p className="ss-muted" style={{ margin: '4px 2px 8px' }}>
                Each point is one trading day&apos;s news centroid; green/red = the index actually
                went up/down. Click a day to open its headline cloud.
              </p>
              <Plot data={allTraces} layout={layout} config={PLOT_CONFIG}
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
              Every headline of the chosen day, projected into the same 16-dim PCA space as the
              model&apos;s <code>embpca</code> features. The blue diamond is the day centroid —
              exactly what the model consumes for that day.
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
                <Plot data={headlineTraces(day, axes)} layout={layout} config={PLOT_CONFIG}
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

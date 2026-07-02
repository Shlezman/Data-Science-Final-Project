import React, { useEffect, useMemo, useState } from 'react';
import { getJson } from '../lib/api.js';
import { Plot, darkLayout, PLOT_CONFIG, UP, DOWN, NEUTRAL } from '../lib/plotly.js';

/**
 * Builds a colored 3D scatter trace for a subset of day-centroids.
 *
 * @param {Array} pts Points [{x,y,z,date,n_headlines,actual}].
 * @param {boolean|null} actual Which class this trace holds (null = unknown).
 * @param {string} name Legend label.
 * @param {string} color Marker color.
 * @returns {object|null} A Plotly trace, or null if empty.
 */
function trace(pts, actual, name, color) {
  const sel = pts.filter((p) => p.actual === actual);
  if (!sel.length) return null;
  return {
    type: 'scatter3d', mode: 'markers', name,
    x: sel.map((p) => p.x), y: sel.map((p) => p.y), z: sel.map((p) => p.z),
    text: sel.map((p) => `${p.date} · ${p.n_headlines} headlines`),
    hovertemplate: '%{text}<extra></extra>',
    marker: { size: 4, color, opacity: 0.85 },
  };
}

/**
 * Right-side sliding drawer with a Plotly 3D scatter of each trading day's
 * news-embedding centroid (leak-safe embpca_000..002), colored green/red by the
 * realised up/down (gray = not yet settled). A slider sweeps forward through
 * time, cumulatively revealing days. Reads /api/centroids once.
 *
 * @returns {JSX.Element} The toggle button + drawer.
 */
export default function Centroids3D() {
  const [points, setPoints] = useState([]);
  const [open, setOpen] = useState(false);
  const [upto, setUpto] = useState(0);

  useEffect(() => {
    getJson('/api/centroids')
      .then((d) => {
        const pts = d?.points || [];
        setPoints(pts);
        setUpto(pts.length ? pts.length - 1 : 0);
      })
      .catch(() => setPoints([]));
  }, []);

  const shown = useMemo(() => points.slice(0, upto + 1), [points, upto]);
  const traces = useMemo(
    () => [
      trace(shown, true, 'Up', UP),
      trace(shown, false, 'Down', DOWN),
      trace(shown, null, 'Unsettled', NEUTRAL),
    ].filter(Boolean),
    [shown],
  );

  const layout = darkLayout({
    showlegend: true,
    legend: { orientation: 'h', y: -0.05 },
    scene: {
      xaxis: { title: 'pca-0', gridcolor: 'rgba(255,255,255,0.1)' },
      yaxis: { title: 'pca-1', gridcolor: 'rgba(255,255,255,0.1)' },
      zaxis: { title: 'pca-2', gridcolor: 'rgba(255,255,255,0.1)' },
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
  });

  return (
    <>
      <button className="ss-drawer__toggle" onClick={() => setOpen(true)}>
        3D centroids
      </button>
      <div className={`ss-drawer ${open ? 'is-open' : ''}`}>
        <div className="ss-drawer__head">
          <span>Daily news centroids (3D)</span>
          <button className="ss-drawer__close" onClick={() => setOpen(false)}>×</button>
        </div>
        {points.length === 0 ? (
          <p className="ss-muted" style={{ padding: '0 16px' }}>
            No centroid data — needs <code>daily_embedding_derived</code> + <code>champion_full_eval</code>.
          </p>
        ) : (
          <div className="ss-drawer__body">
            <Plot data={traces} layout={layout} config={PLOT_CONFIG}
                  style={{ width: '100%', height: '70vh' }} useResizeHandler />
            <div className="ss-drawer__slider">
              <input type="range" min={0} max={points.length - 1} value={upto}
                     onChange={(e) => setUpto(Number(e.target.value))} />
              <span className="ss-muted">
                through {points[upto]?.date} · {shown.length}/{points.length} days
              </span>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
